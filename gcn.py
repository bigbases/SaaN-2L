import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN, LinkEmbedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt


class GCN():
    def node_classification(G, node_subjects, args):
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

        # Creating the GCN layers
        generator = FullBatchNodeGenerator(G, method="gcn")
        train_gen = generator.flow(train_subjects.index, train_targets)

        gcn = GCN(
        layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.5
        )

        x_inp, x_out = gcn.in_out_tensors()
        predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

        # Training the model
        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(
            optimizer=optimizers.Adam(lr=0.01),
            loss=losses.categorical_crossentropy,
            metrics=["acc"],
        )

        val_gen = generator.flow(val_subjects.index, val_targets)
        es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)

        history = model.fit(
            train_gen,
            epochs=200,
            validation_data=val_gen,
            verbose=0,
            shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
            callbacks=[es_callback],
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
        all_gen = generator.flow(all_nodes)

        # Node embeddings
        embedding_model = Model(inputs=x_inp, outputs=x_out)
        emb = embedding_model.predict(all_gen)

        X = emb.squeeze(0)
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
        return X

    def link_prediction(G, args):
        # Define an edge splitter on the original graph G
        edge_splitter_test = EdgeSplitter(G)

        # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
        # reduced graph G_test with the sampled links removed:
        G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
            p=0.1, method="global", keep_connected=True)

        # Define an edge splitter on the reduced graph G_test:
        edge_splitter_train = EdgeSplitter(G_test)

        # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
        # reduced graph G_train with the sampled links removed:
        G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
            p=0.1, method="global", keep_connected=True)

        # Creating the GCN link model
    
        print("Training Start!")
        start = time.time()

        epochs = 50
        train_gen = FullBatchLinkGenerator(G_train, method="gcn")
        train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

        test_gen = FullBatchLinkGenerator(G_test, method="gcn")
        test_flow = train_gen.flow(edge_ids_test, edge_labels_test)

        gcn = GCN(
            layer_sizes=[16, 16], activations=["relu", "relu"], generator=train_gen, dropout=0.3)

        x_inp, x_out = gcn.in_out_tensors()
        prediction = LinkEmbedding(activation="relu", method="ip")(x_out)
        prediction = keras.layers.Reshape((-1,))(prediction)

        model = keras.Model(inputs=x_inp, outputs=prediction)

        model.compile(
            optimizer=keras.optimizers.Adam(lr=0.01),
            loss=keras.losses.binary_crossentropy,
            metrics=["acc"],
        )

        history = model.fit(
            train_flow, epochs=epochs, validation_data=test_flow, verbose=0, shuffle=False
        )
        print("\nTraining Done! (Time: ",time.time() - start, ")")

        # Plotting
        sg.utils.plot_history(history)

        train_metrics = model.evaluate(train_flow)
        test_metrics = model.evaluate(test_flow)

        print("Evaluate the baseline performance on link prediction task")

        print("\nTest Set Metrics of the trained model:")
        for name, val in zip(model.metrics_names, test_metrics):
            print("\t{}: {:0.4f}".format(name, val))


    def subgraph_learning(subgraphList, args):
        subgraph = ig.induced_subgraph(subgraphList,implementation="create_from_scratch")
        fea_mat_temp = fea_mat[fea_mat.index.isin(subgraph.vs['_nx_name'])] # subgraph들의 feature 추출
        
        subgraph_ = StellarGraph.from_networkx(subgraph.to_networkx(), node_features = fea_mat_temp.reset_index(drop=True))
        
        generator = FullBatchNodeGenerator(subgraph_, method="gcn")
        
        subnode_subjects = node_subjects[node_subjects.index.isin(subgraph.vs['_nx_name'])].reset_index(drop=True) # subgraph들의 라벨
        
        # Split
        train_subjects, test_subjects = model_selection.train_test_split(
        subnode_subjects, train_size=0.7, test_size=None
        )
        val_subjects, test_subjects = model_selection.train_test_split(
            subnode_subjects, train_size=0.5, test_size=None
        )
        
        target_encoding = preprocessing.LabelBinarizer()
        train_targets = target_encoding.fit_transform(train_subjects)
        val_targets = target_encoding.transform(val_subjects)
        test_targets = target_encoding.transform(test_subjects)
        
        train_gen = generator.flow(train_subjects.index, train_targets)
        
        gcn = GCN(
        layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.5
        )
        
        
        x_inp, x_out = gcn.in_out_tensors()
        predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.01),
            loss=losses.categorical_crossentropy,
            metrics=["acc"],
        )   
        
        val_gen = generator.flow(val_subjects.index, val_targets)

        es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)

        history = model.fit(
            train_gen,
            epochs=100,
            validation_data=val_gen,
            verbose=0,
            shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
            callbacks=[es_callback],
        )

        all_nodes = subnode_subjects.index
        all_gen = generator.flow(all_nodes)
        all_targets = target_encoding.fit_transform(node_subjects)

        embedding_model = Model(inputs=x_inp, outputs=x_out)
        
        emb = embedding_model.predict(all_gen)

        return emb.squeeze(0)
