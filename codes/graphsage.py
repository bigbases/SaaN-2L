class GraphSAGE():
    
    def node_classification(G, node_subjects, args):
        #Global parameters
        epochs = 20  
        train_size = 0.2
        test_size = 0.15
        val_size = 0.2
        batch_size = 50
        num_samples = [10, 5]

        # Splitting the data
        train_subjects, test_subjects = model_selection.train_test_split(node_subjects, train_size=train_size, test_size=None, stratify=node_subjects)
        val_subjects, test_subjects = model_selection.train_test_split(test_subjects, train_size=test_size, test_size=None, stratify=test_subjects)

        # Converting to numeric arrays
        target_encoding = preprocessing.LabelBinarizer()
        train_targets = target_encoding.fit_transform(train_subjects)
        val_targets = target_encoding.transform(val_subjects)
        test_targets = target_encoding.transform(test_subjects)

        print("Training Start!")
        start = time.time()

        # Creating the GraphSAGE model
        generator = GraphSAGENodeGenerator(G, batch_size, num_samples)
        train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)

        graphsage_model = GraphSAGE(
            layer_sizes=[32, 32], generator=generator, bias=True, dropout=0.5,
        )

        x_inp, x_out = graphsage_model.in_out_tensors()
        prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)
        
        # Training the model
        model = Model(inputs=x_inp, outputs=prediction)
        model.compile(
            optimizer=optimizers.Adam(lr=0.005),
            loss=losses.categorical_crossentropy,
            metrics=["acc"],
        )

        history = model.fit(
            train_gen, epochs=20, validation_data=test_gen, verbose=2, shuffle=False
        )
        print("\nTraining Done! (Time: ",time.time() - start, ")")

        # Plotting
        sg.utils.plot_history(history)

        # Evaluating
        test_gen = generator.flow(test_subjects.index, test_targets)
        test_metrics = model.evaluate(test_gen)
        print("\nTest Set Metrics:")
        for name, val in zip(model.metrics_names, test_metrics):
            print("\t{}: {:0.4f}".format(name, val))

        # Making predictions with the model
        all_nodes = node_subjects.index
        all_mapper = generator.flow(all_nodes)
        all_predictions = model.predict(all_mapper)
        node_predictions = target_encoding.inverse_transform(all_predictions)

        # Node embeddings
        embedding_model = Model(inputs=x_inp, outputs=x_out)
        emb = embedding_model.predict(all_mapper)

        X = emb
        y = np.argmax(target_encoding.transform(node_subjects), axis=1)

        # Baseline Performance Evaluation
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)
        model_emb = keras.models.Sequential()
        model_emb.add(layers.Dense(train_targets.shape[1], activation='softmax', input_shape=(16,)))
        model_emb.compile(keras.optimizers.Adam(learning_rate=0.01), 
                    loss=keras.losses.categorical_crossentropy,
                    metrics=['accuracy'])

        start = time.time()
        history = model_emb.fit(X_train,
                            y_train,
                            epochs=50,
                            verbose=0,
                            shuffle=False,
                            batch_size=128)
        print(time.time() - start)

        # Evaluate the baseline model on the test data 
        print("Evaluate the baseline performance on node classification task")
        results = model_emb.evaluate(X_test, y_test, batch_size=128)
        print("test loss, test acc:", results)


    def link_prediction(G, args):
        # Define an edge splitter on the original graph G
        edge_splitter_test = EdgeSplitter(G)

        # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
        # reduced graph G_test with the sampled links removed:
        G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
            p=0.1, method="global", keep_connected=True
        )

        # Define an edge splitter on the reduced graph G_test:
        edge_splitter_train = EdgeSplitter(G_test)

        # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
        # reduced graph G_train with the sampled links removed:
        G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
            p=0.1, method="global", keep_connected=True
        )

        print("Training Start!")
        start = time.time()

        train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples, weighted =True)
        train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)
        test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples, weighted =True)
        test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

        graphsage = GraphSAGE(
            layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.3
        )

        # Build the model and expose input and output sockets of graphsage model for link prediction
        x_inp, x_out = graphsage.in_out_tensors()

        prediction = link_classification(
            output_dim=1, output_act="relu", edge_embedding_method="ip"
        )(x_out)

        model = keras.Model(inputs=x_inp, outputs=prediction)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.binary_crossentropy,
            metrics=["acc"],
        )

        history = model.fit(train_flow, epochs=10, validation_data=test_flow, verbose=2)
        
        print("\nTraining Done! (Time: ",time.time() - start, ")")

        # Plotting
        sg.utils.plot_history(history)

        train_metrics = model.evaluate(train_flow)
        test_metrics = model.evaluate(test_flow)

        print("Evaluate the baseline performance on link prediction task")

        print("\nTest Set Metrics of the trained model:")
        for name, val in zip(model.metrics_names, test_metrics):
            print("\t{}: {:0.4f}".format(name, val))

        x_inp_src = x_inp[0::2]
        x_out_src = x_out[0]
        embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
        
        node_ids = G.nodes()
        node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(node_ids)
        node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)
        
        return node_embeddings

    def subgraph_learning(subgraphList, args):
        subgraph = ig.induced_subgraph(subgraphList,implementation="create_from_scratch")
        subnode_features = node_features[node_features.index.isin(subgraph.vs['_nx_name'])] # subgraph들의 feature 추출

        subgraph_ = StellarGraph.from_networkx(subgraph.to_networkx(), node_features = subnode_features.reset_index(drop=True))
    
        subnodes = list(subgraph_.nodes())
        sub_unsupervised_samples = UnsupervisedSampler(
            subgraph_, nodes = subnodes, length=5, number_of_walks=1
        )
        
        sub_generator = GraphSAGELinkGenerator(subgraph_, batch_size, num_samples)
        sub_train_gen = sub_generator.flow(sub_unsupervised_samples)
        
        sub_graphsage = GraphSAGE(
            layer_sizes = layer_sizes, generator=sub_generator, bias=True, dropout=0.0, normalize="l2"
        )
        
        x_inp, x_out = sub_graphsage.in_out_tensors()
        x_inp_src = x_inp[0::2]
        x_out_src = x_out[0]
        sub_embedding_model = keras.Model(inputs = x_inp_src, outputs = x_out_src)
        
        sub_node_ids = subgraph_.nodes()
        sub_node_gen = GraphSAGENodeGenerator(subgraph_, batch_size, num_samples).flow(sub_node_ids)
        
        sub_node_embeddings = sub_embedding_model.predict(sub_node_gen, workers=4, verbose=1)
        
        return sub_node_embeddings

    