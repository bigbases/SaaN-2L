import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score

import os
import networkx as nx
import numpy as np
import pandas as pd
from tensorflow import keras

from stellargraph import StellarGraph, datasets
from stellargraph.data import UnsupervisedSampler
from stellargraph.data import BiasedRandomWalk
from stellargraph.mapper import Node2VecLinkGenerator, Node2VecNodeGenerator
from stellargraph.layer import Node2Vec, link_classification
from stellargraph.data import EdgeSplitter
import multiprocessing
from gensim.models import Word2Vec

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


class Node2Vec():
    def node_classification(G, node_subjects, args):
        walk_number = 100
        walk_length = 5
        batch_size = 50
        epochs = 2
        emb_size = 128

        walker = BiasedRandomWalk(
            G,
            n=walk_number,
            length=walk_length,
            p=1.0,
            q=1.0
        )

        unsupervised_samples = UnsupervisedSampler(G, nodes=list(G.nodes()), walker=walker)
        print("Training Start!")
        start = time.time()

        generator = Node2VecLinkGenerator(G, batch_size)
        node2vec = Node2Vec(emb_size, generator=generator)

        x_inp, x_out = node2vec.in_out_tensors()

        prediction = link_classification(
            output_dim=1, output_act="sigmoid", edge_embedding_method="dot"
        )(x_out)

        model = keras.Model(inputs=x_inp, outputs=prediction)

        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss=keras.losses.binary_crossentropy,
            metrics=[keras.metrics.binary_accuracy],
        )

        history = model.fit(
            generator.flow(unsupervised_samples),
            epochs=epochs,
            verbose=1,
            use_multiprocessing=False,
            workers=4,
            shuffle=True,
        )
        print("\nTraining Done! (Time: ",time.time() - start, ")")
        
        # Node embeddings
        x_inp_src = x_inp[0]
        x_out_src = x_out[0]
        embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

        node_gen = Node2VecNodeGenerator(G, batch_size).flow(subjects.index)
        node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

        # X will hold the 128-dimensional input features
        X = node_embeddings
        # y holds the corresponding target values
        y = np.array(subjects)

        # Baseline Performance Evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, test_size=None)

        clf = LogisticRegressionCV(
            Cs=10, cv=10, scoring="accuracy", verbose=False, multi_class="ovr", max_iter=300
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        # Evaluate the baseline model on the test data 
        accuracy_score(y_test, y_pred)

    def link_prediction(G, args):
        # Define an edge splitter on the original graph:
        edge_splitter_test = EdgeSplitter(graph)

        # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
        # reduced graph graph_test with the sampled links removed:
        graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(
            p=0.1, method="global"
        )

        # Do the same process to compute a training subset from within the test graph
        edge_splitter_train = EdgeSplitter(graph_test, graph)
        graph_train, examples, labels = edge_splitter_train.train_test_split(
            p=0.1, method="global"
        )
        (
            examples_train,
            examples_model_selection,
            labels_train,
            labels_model_selection,
        ) = train_test_split(examples, labels, train_size=0.75, test_size=0.25)

        p = 1.0
        q = 1.0
        dimensions = 128
        num_walks = 10
        walk_length = 80
        window_size = 10
        num_iter = 1
        workers = multiprocessing.cpu_count()


        def node2vec_embedding(graph, name):
            rw = BiasedRandomWalk(graph)
            walks = rw.run(graph.nodes(), n=num_walks, length=walk_length, p=p, q=q)
            print(f"Number of random walks for '{name}': {len(walks)}")

            model = Word2Vec(
                walks,
                size=dimensions,
                window=window_size,
                min_count=0,
                sg=1,
                workers=workers,
                iter=num_iter,
            )

            def get_embedding(u):
                return model.wv[u]

            return get_embedding
        
        print("Training Start!")
        start = time.time()

        embedding_train = node2vec_embedding(graph_train, "Train Graph")

        ###### HELPER FUNCTION ######

        # 1. link embeddings
        def link_examples_to_features(link_examples, transform_node, binary_operator):
            return [
                binary_operator(transform_node(src), transform_node(dst))
                for src, dst in link_examples
            ]


        # 2. training classifier
        def train_link_prediction_model(
            link_examples, link_labels, get_embedding, binary_operator
        ):
            clf = link_prediction_classifier()
            link_features = link_examples_to_features(
                link_examples, get_embedding, binary_operator
            )
            clf.fit(link_features, link_labels)
            return clf


        def link_prediction_classifier(max_iter=2000):
            lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
            return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


        # 3. and 4. evaluate classifier
        def evaluate_link_prediction_model(
            clf, link_examples_test, link_labels_test, get_embedding, binary_operator
        ):
            link_features_test = link_examples_to_features(
                link_examples_test, get_embedding, binary_operator
            )
            score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
            return score


        def evaluate_roc_auc(clf, link_features, link_labels):
            predicted = clf.predict_proba(link_features)

            # check which class corresponds to positive links
            positive_column = list(clf.classes_).index(1)
            return roc_auc_score(link_labels, predicted[:, positive_column])

        def operator_hadamard(u, v):
            return u * v


        def operator_l1(u, v):
            return np.abs(u - v)


        def operator_l2(u, v):
            return (u - v) ** 2


        def operator_avg(u, v):
            return (u + v) / 2.0


        def run_link_prediction(binary_operator):
            clf = train_link_prediction_model(
                examples_train, labels_train, embedding_train, binary_operator
            )
            score = evaluate_link_prediction_model(
                clf,
                examples_model_selection,
                labels_model_selection,
                embedding_train,
                binary_operator,
            )

            return {
                "classifier": clf,
                "binary_operator": binary_operator,
                "score": score,
            }


        binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]

        results = [run_link_prediction(op) for op in binary_operators]
        best_result = max(results, key=lambda result: result["score"])

        print(f"Best result from '{best_result['binary_operator'].__name__}'")

        pd.DataFrame(
            [(result["binary_operator"].__name__, result["score"]) for result in results],
            columns=("name", "ROC AUC score"),
        ).set_index("name")


        embedding_test = node2vec_embedding(graph_test, "Test Graph")

        test_score = evaluate_link_prediction_model(
            best_result["classifier"],
            examples_test,
            labels_test,
            embedding_test,
            best_result["binary_operator"],
        )
        print(
            f"ROC AUC score on test set using '{best_result['binary_operator'].__name__}': {test_score}"
        )

    def subgraph_learning(subgraphList, args):
        subgraph = ig.induced_subgraph(subgraphList,implementation="create_from_scratch")
        isin_filter = node_features_encoded['userID'].isin(subgraph.vs['id'])
        
        subgraph_features = node_features_encoded[isin_filter]
        subgraph_country_degree = pd.concat([subgraph_features['countrycode_encoded'], subgraph_features['degree']],axis=1)
        subgraph_country_degree.reset_index(drop=True,inplace=True)
        
        subgraph_ = StellarGraph.from_networkx(subgraph.to_networkx(), node_type_default = "user", edge_type_default = "friendship", node_features = subgraph_country_degree)
        
        rw = BiasedRandomWalk(subgraph_)
        walks = rw.run(subgraph_.nodes(), n=num_walks, length=walk_length, p=p, q=q)

        model = Word2Vec(
            walks,
            vector_size =dimensions,
            window=window_size,
            min_count=0,
            sg=1,
            workers=workers,
            epochs=num_iter,
        )

        return model.wv

    