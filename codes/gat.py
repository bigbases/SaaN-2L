import networkx as nx
import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class GAT():
    def node_classification(G):
        # Splitting the data
        train_subjects, test_subjects = model_selection.train_test_split(
            node_subjects, train_size=140, test_size=None, stratify=node_subjects
        )
        val_subjects, test_subjects = model_selection.train_test_split(
            test_subjects, train_size=500, test_size=None, stratify=test_subjects
        )

        # Converting to numeric arrays
        target_encoding = preprocessing.LabelBinarizer()
        train_targets = target_encoding.fit_transform(train_subjects)
        val_targets = target_encoding.transform(val_subjects)
        test_targets = target_encoding.transform(test_subjects)

        print("Training Start!")
        start = time.time()

        # Creating the GAT layers
        generator = FullBatchNodeGenerator(G, method="gat")
        train_gen = generator.flow(train_subjects.index, train_targets)

        gat = GAT(
            layer_sizes=[8, train_targets.shape[1]],
            activations=["elu", "softmax"],
            attn_heads=8,
            generator=generator,
            in_dropout=0.5,
            attn_dropout=0.5,
            normalize=None,
        )

        # Training the model
        x_inp, predictions = gat.in_out_tensors()
        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(
            optimizer=optimizers.Adam(lr=0.005),
            loss=losses.categorical_crossentropy,
            metrics=["acc"],
        )

        val_gen = generator.flow(val_subjects.index, val_targets)

        if not os.path.isdir("logs"):
            os.makedirs("logs")
        es_callback = EarlyStopping(
            monitor="val_acc", patience=20
        )  # patience is the number of epochs to wait before early stopping in case of no further improvement
        mc_callback = ModelCheckpoint(
            "logs/best_model.h5", monitor="val_acc", save_best_only=True, save_weights_only=True
        )

        history = model.fit(
            train_gen,
            epochs=50,
            validation_data=val_gen,
            verbose=2,
            shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
            callbacks=[es_callback, mc_callback],
        )
        print("\nTraining Done! (Time: ",time.time() - start, ")")

        # Plotting
        sg.utils.plot_history(history)
        model.load_weights("logs/best_model.h5")

        # Evaluating
        test_gen = generator.flow(test_subjects.index, test_targets)
        test_metrics = model.evaluate(test_gen)
        print("\nTest Set Metrics:")
        for name, val in zip(model.metrics_names, test_metrics):
            print("\t{}: {:0.4f}".format(name, val))

        # Making predictions with the model
        all_nodes = node_subjects.index
        all_gen = generator.flow(all_nodes)
        all_predictions = model.predict(all_gen)

        # Node embeddings
        node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
        emb_layer = next(l for l in model.layers if l.name.startswith("graph_attention"))
        embedding_model = Model(inputs=x_inp, outputs=emb_layer.output)
        emb = embedding_model.predict(all_gen)

        X = emb.squeeze()
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


    def subgraph_learning(subgraphList):
        subgraph = ig.induced_subgraph(subgraphList,implementation="create_from_scratch")
        fea_mat_temp = fea_mat[fea_mat.index.isin(subgraph.vs['_nx_name'])] # subgraph들의 feature 추출
        
        subgraph_ = StellarGraph.from_networkx(subgraph.to_networkx(), node_features = fea_mat_temp.reset_index(drop=True))
        
        generator = FullBatchNodeGenerator(subgraph_, method="gat")
        
        node_subjects = labels[labels.index.isin(subgraph.vs['_nx_name'])].reset_index(drop=True) # subgraph들의 라벨
        
        # Split
        train_subjects, test_subjects = model_selection.train_test_split(
        node_subjects, train_size=0.7, test_size=None
        )
        val_subjects, test_subjects = model_selection.train_test_split(
            node_subjects, train_size=0.5, test_size=None
        )
        
        target_encoding = preprocessing.LabelBinarizer()
        train_targets = target_encoding.fit_transform(train_subjects)
        val_targets = target_encoding.transform(val_subjects)
        test_targets = target_encoding.transform(test_subjects)
        
        train_gen = generator.flow(train_subjects.index, train_targets)
        
        gat = GAT(
        layer_sizes=[8, train_targets.shape[1]],
        activations=["elu", "softmax"],
        attn_heads=8,
        generator=generator,
        in_dropout=0.5,
        attn_dropout=0.5,
        normalize=None,
        )

        x_inp, predictions = gat.in_out_tensors()
        
        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.005),
            loss=losses.categorical_crossentropy,
            metrics=["acc"],
        )
        
        val_gen = generator.flow(val_subjects.index, val_targets)
        if not os.path.isdir("logs"):
            os.makedirs("logs")
        es_callback = EarlyStopping(
            monitor="val_acc", patience=20
        )  # patience is the number of epochs to wait before early stopping in case of no further improvement
        mc_callback = ModelCheckpoint(
            "logs/best_model.h5", monitor="val_acc", save_best_only=True, save_weights_only=True
        )

        history = model.fit(
            train_gen,
            epochs=100,
            validation_data=val_gen,
            verbose=0,
            shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
            callbacks=[es_callback, mc_callback],
        )

        all_nodes = node_subjects.index
        all_gen = generator.flow(all_nodes)
        all_targets = target_encoding.fit_transform(node_subjects)
        
        emb_layer = next(l for l in model.layers if l.name.startswith("graph_attention"))
        embedding_model = Model(inputs=x_inp, outputs=emb_layer.output)

        emb = embedding_model.predict(all_gen)

        return emb.squeeze(0)
        