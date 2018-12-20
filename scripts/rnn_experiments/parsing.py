import os.path as p


class Annotation(object):
    def __init__(self, sentence_index, sentence_tokens, interval):
        self.sentence_index = sentence_index
        self.sentence_tokens = sentence_tokens
        self.start, self.end = interval

        # Get the tokens
        self.annotation_tokens = self.sentence_tokens[self.start:self.end+1]

        
class EventAnnotation(Annotation):
    def __init__(self, sentence_index, sentence_tokens, interval, annotated_contexts=None):
        super(EventAnnotation, self).__init__(sentence_index, sentence_tokens, interval)
        self.contexts = annotated_contexts

    @staticmethod
    def from_annotated_event(sentences, row_string):
        # Parse the row
        elems = row_string.split()
        sen_ix = int(elems[0])
        start, end = (int(i) for i in elems[1].split('-'))
        annotated_contexts = elems[2].split(',')

        # Select the sentence
        sentence = sentences[sen_ix].split()

        # Return the created instance
        return EventAnnotation(sen_ix, sentence, (start, end), annotated_contexts)

    def get_evtid(self):
        return "E%i_%i_%i" % (self.sentence_index, self.start, self.end)


class ContextAnnotation(Annotation):
    def __init__(self, sentence_index, sentence_tokens, interval, gid):
        super(ContextAnnotation, self).__init__(sentence_index, sentence_tokens, interval)
        self.gid = gid

    @staticmethod
    def from_annotated_mentions(sentences, row_string):
        # Parse the row
        elems = row_string.split()
        sen_ix, raw_data = int(elems[0]), elems[1]
        data = raw_data.split("%")
        start, end = int(data[0]), int(data[1])
        gid = data[3]

        # Select the sentence
        sentence = sentences[sen_ix].split()

        # Return the created instance
        return ContextAnnotation(sen_ix, sentence, (start, end), gid)

    @staticmethod
    def from_manually_annotated_mentions(sentences, row_string):
        # Parse the row
        elems = row_string.split()
        sen_ix = int(elems[0])
        start, end = (int(i) for i in elems[1].split('-'))
        gid = elems[2]

        # Select the sentence
        sentence = sentences[sen_ix].split()

        # Return the created instance
        return ContextAnnotation(sen_ix, sentence, (start, end), gid)


def parse_directory(path):
    # Parses the data from a paper directory
    with open(p.join(path,  "sentences.txt")) as f:
        sentences = [l.strip().lower() for l in f.readlines()]

    with open(p.join(path, "annotated_event_intervals.tsv")) as f:
        events = [EventAnnotation.from_annotated_event(sentences, l.strip()) for l in f.readlines() if len(l.strip().split('\t')) > 2]

    with open(p.join(path, "mention_intervals.txt")) as f:
        mentions = [ContextAnnotation.from_annotated_mentions(sentences, l.strip()) for l in f.readlines() if len(l.strip().split()) > 1]

    with open(p.join(path, "manual_context_mentions.tsv")) as f:
        mentions += [ContextAnnotation.from_manually_annotated_mentions(sentences, l.strip()) for l in f.readlines()]

    return sentences, events, mentions