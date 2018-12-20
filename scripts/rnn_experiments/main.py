from parsing import *
from glob import glob
import os.path as path
import itertools as it
from collections import namedtuple
import dynet as dy
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import sys
import os
import utils.utils as utils

ParsedData = namedtuple('ParsedData', 'sentences, events, mentions')


def find_closest_context(parsed_data, evtid, ctxid):

    try:
        event = parsed_data.events[evtid]
    except Exception as e:
        print("Caused by EvtID: {}".format(evtid))
    try:
        ctxs = parsed_data.mentions[ctxid]
    except Exception as e:
        print("Caused by CtxID: {}".format(ctxid))
    evt_ix = event.sentence_index
    sorted_mentions = sorted(ctxs, key=lambda c: abs(evt_ix - c.sentence_index))
    # TODO: Maybe untie here with the token offsets
    closest_ctx = sorted_mentions[0]
    ctx_ix = closest_ctx.sentence_index

    start, end = min(evt_ix, ctx_ix), max(evt_ix, ctx_ix)
    # TODO: Cut the tokens up until end of the event and context tokens
    sentences = parsed_data.sentences[start:end+1]

    return event, closest_ctx, sentences

def run_instance(instance, builder, wemb, ix, W, V, b):

    # Renew the computational graph
    #dy.renew_cg()

    # Fetch the embeddings for the current sentence
    words = instance
    inputs = [wemb[ix[w]] for w in words]

    # Run FF over the LSTM
    lstm = builder.initial_state()
    outputs = lstm.transduce(inputs)

    # Get the last embedding
    selected = outputs[-1]

    # Run the FF network for classification
    prediction = dy.logistic(V * (W * selected + b))

    return prediction


#def prediction_loss(instance, prediction):
#    # Compute the loss
#    y_true = dy.scalarInput(1 if instance.polarity else 0)
#    loss = dy.binary_log_loss(prediction, y_true)
#
#    return loss


if __name__ == "__main__":
    dir = "../../data/papers/"

    #relevant_papers = ['PMC3032653', 'PMC2064697', 'PMC3717960', 'PMC3233644',
    #   'PMC2195994', 'PMC3461631', 'PMC3058384', 'PMC534114',
    #   'PMC2063868', 'PMC2193052', 'PMC2773002', 'PMC3378484',
    #   'PMC2978746', 'PMC2156142', 'PMC420486', 'PMC3190874',
    #   'PMC2196001', 'PMC2743561', 'PMC4204162', 'PMC3203906',
    #   'PMC3392545', 'PMC3135394', 'PMC4052680']

    data = dict()
    for d in glob(path.join(dir, "PMC*")):
        key = d.split(path.sep)[-1]
        sentences, events, mentions = parse_directory(d)
        evts = {e.get_evtid(): e for e in events}
        ctxts = {gid:set(values) for gid, values in it.groupby(sorted(mentions, key = lambda m: m.gid), lambda m: m.gid)}
        data[key] = ParsedData._make((sentences, evts, ctxts))

    vocabulary = {w:i for i, w in enumerate(set(it.chain.from_iterable(s.split() for s in it.chain.from_iterable(v.sentences for v in data.values()))))}

    # ==============================================================================
    # BEGIN PREAMBLE
    # ==============================================================================
    print("Beginning preamble...")
    json_config = utils.Config("../../config.json")
    data_path = json_config.get_features_filepath()

    # Load the Grouped DataFrame
    groups_path = os.path.join(data_path, "grouped_features.csv")
    df = utils.load_dataframe(groups_path)

    # Remove degenerate paper
    gdf2 = df[df.PMCID != "b'PMC4204162'"]

    # Load train/validate/test folds
    fold_path = os.path.join(json_config.get_features_filepath(), "cv_folds_val_4.pkl")
    cv_folds = utils.load_cv_folds(fold_path)
    cv_folds = [(train+validate, test) for train, validate, test in cv_folds]

    print("Finished preamble.")
    # ==============================================================================
    # END PREAMBLE
    # ==============================================================================


    lstm_data = dict()
    lstm_labels = dict()
    failures = 0
    for index, row in gdf2.iterrows():
        pmcid, evtid, ctxid = row.PMCID[2:-1], row.EvtID[2:-1], row.CtxID[2:-1]
        label = row.label
        parsed_data = data[pmcid]
        try:
            key = (pmcid, evtid, ctxid)
            event, closest_ctx, sentences = find_closest_context(parsed_data, evtid, ctxid)
            tokens = [t for s in sentences for t in s.split()]
            lstm_data[key] = tokens
            lstm_labels[key] = label
        except:
            print("^Find error in ", pmcid)
            # sys.exit()
            failures += 1

    print("Failes %i out of %i" % (failures, gdf2.shape[0]))
    sys.exit()
    VOC_SIZE = len(vocabulary)
    WEM_DIMENSIONS = 50

    NUM_LAYERS = 1
    HIDDEN_DIM = 20

    FF_HIDDEN_DIM = 10

    print("Vocabulary size: %i" % VOC_SIZE)

    params = dy.ParameterCollection()
    wemb = params.add_lookup_parameters((VOC_SIZE, WEM_DIMENSIONS))
    # Feed-Forward parameters
    W = params.add_parameters((FF_HIDDEN_DIM, HIDDEN_DIM))
    b = params.add_parameters(FF_HIDDEN_DIM)
    V = params.add_parameters((1, FF_HIDDEN_DIM))

    builder = dy.LSTMBuilder(NUM_LAYERS, WEM_DIMENSIONS, HIDDEN_DIM, params)

    # CV loop
    for train_ix, test_ix in cv_folds:
        keys_train = [(r.PMCID[2:-1], r.EvtID[2:-1], r.CtxID[2:-1]) for _, r in gdf2.iloc[train_ix].iterrows()]
        train = [lstm_data[k] for k in keys_train if k in lstm_data]
        train_labels = [lstm_labels[k] for k in keys_train if k in lstm_data]

        keys_test = [(r.PMCID[2:-1], r.EvtID[2:-1], r.CtxID[2:-1]) for _, r in gdf2.iloc[test_ix].iterrows()]
        test = [lstm_data[k] for k in keys_test if k in lstm_data]
        test_labels = [lstm_labels[k] for k in keys_test if k in lstm_data]

        # Training loop
        trainer = dy.SimpleSGDTrainer(params)
        epochs = 5
        for e in range(epochs):
            # Shuffle the training instances
            dy.renew_cg()
            training_losses = list()
            for i, (instance, label) in enumerate(tqdm(list(zip(train, train_labels)), desc="Training")):
                prediction = run_instance(instance, builder, wemb, vocabulary, W, V, b)

                y_true = dy.scalarInput(1 if label else 0)
                loss = dy.binary_log_loss(prediction, y_true)   # prediction_loss(instance, prediction)

                #loss.backward()
                #trainer.update()

                #loss_value = loss.value()
                training_losses.append(loss)

                if i % 200 == 0 and i != 0:
                    avg_loss = dy.esum(training_losses) / len(training_losses)
                    training_losses = list()

                    avg_loss.backward()
                    trainer.update()

            #avg_loss = np.average(training_losses)
            dy.renew_cg()
            # Now do testing
            testing_losses = list()
            testing_predictions = list()
            for i, (instance, label) in enumerate(tqdm(list(zip(test, test_labels)), desc="Testing")):
                prediction = run_instance(instance, builder, wemb, vocabulary, W, V, b)
                y_pred = 1 if prediction.value() >= 0.5 else 0
                testing_predictions.append(y_pred)
                y_true = dy.scalarInput(1 if label else 0)
                loss = dy.binary_log_loss(prediction, y_true)   # prediction_loss(instance, prediction)
                loss_value = loss.value()
                testing_losses.append(loss_value)

            f1 = f1_score(test_labels, testing_predictions)
            precision = precision_score(test_labels, testing_predictions)
            recall = recall_score(test_labels, testing_predictions)

            print("Epoch %i average training loss: %f\t average testing loss: %f" % (
            e + 1, np.average(training_losses), np.average(testing_losses)))
            print("Precision: %f\tRecall: %f\tF1: %f" % (precision, recall, f1))
            if sum(testing_predictions) >= 1:
                report = classification_report(test_labels, testing_predictions)
                print(report)
            if avg_loss <= 3e-3:
                break
            print()
