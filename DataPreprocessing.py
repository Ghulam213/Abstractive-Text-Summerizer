try:
    import numpy as np
    import pandas as pd
    import string
    from sklearn.model_selection import train_test_split
    from nltk.corpus import stopwords
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    import nltk
    import pickle
except ImportError as e:
    print(e)


class DataPreprocessing():
    def __init__(self):
        self.max_stories_len = 700
        self.max_highlights_len = 38
        self.reverse_target_word_index = ''
        self.reverse_source_word_index = ''
        # self.target_word_index = ''      if end start tokens are added uncomment it.

    def load_dataset(self, filename, path=''):
        try:
            return pd.read_excel(path + filename + '.xlsx')
        except FileNotFoundError as e:
            print(e)

    def clean_stories(self, lines):
        stop_words = set(stopwords.words('english'))
        cleaned = list()
        # prepare a translation table to remove punctuation
        table = str.maketrans('', '', string.punctuation)
        for line in lines:
            # strip source cnn office if it exists
            index = line.find('(CNN) -- ')
            if index > -1:
                line = line[index + len('(CNN)'):]
            # tokenize on white space
            line = line.split()
            # convert to lower case
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [w.translate(table) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            line = [w for w in line if w not in stop_words]
            # store as string
            cleaned.append(' '.join(line))
        # remove empty strings
        cleaned = [c for c in cleaned if len(c) > 0]
        cleaned = ' '.join(c for c in cleaned)
        return cleaned

    def clean_highlights(self, line):
        stop_words = set(stopwords.words('english'))
        cleaned = list()
        # prepare a translation table to remove punctuation
        table = str.maketrans('', '', string.punctuation)
        # strip source cnn office if it exists
        index = line.find('NEW: ')
        if index > -1:
            line = line[index + len('NEW:'):]
        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation from each token
        line = [w.translate(table) for w in line]
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        line = [w for w in line if w not in stop_words]
        # store as string
        cleaned.append(' '.join(line))
        # remove empty strings
        cleaned = [c for c in cleaned if len(c) > 0]
        return cleaned[0]

    def drop_dulp_and_na(self, df, columns, do_inplace=True):
        for subset in columns:
            df.drop_duplicates(subset=[subset], inplace=do_inplace)  # dropping duplicates
        df.replace('', np.nan, inplace=True)
        df.dropna(axis=0, inplace=do_inplace)
        return df

    def clear_long_text(self, df):
        stories = np.array(df['Stories'])
        highlights = np.array(df['Highlights'])

        short_text = []
        short_summary = []

        for i in range(len(highlights)):
            if len(highlights[i].split()) <= self.max_highlights_len and len(
                    stories[i].split()) <= self.max_stories_len:
                short_text.append(stories[i])
                short_summary.append(highlights[i])

        return pd.DataFrame({'Stories': short_text, 'Highlights': short_summary})

    def start_end_token(self, data):
        data = data.apply(lambda x: 'sostok ' + x + ' eostok')

    def split_data(self, X, y, train_ratio, dev_ratio, random=0, do_shuffle=True):
        X_tr, X_test, y_tr, y_test = train_test_split(np.array(X), np.array(y), test_size=(1 - train_ratio),
                                                      random_state=random, shuffle=do_shuffle)
        dev_len = int(dev_ratio * len(X))
        X_dev, X_test, y_dev, y_test = X_test[:dev_len], X_test[dev_len:], y_test[:dev_len], y_test[dev_len:]

        return X_tr, X_test, X_dev, y_tr, y_test, y_dev

    def rare_words_count(self, data, thresh=5):
        data_tokenizer = Tokenizer()
        data_tokenizer.fit_on_texts(list(data))

        cnt = 0
        tot_cnt = 0
        freq = 0
        tot_freq = 0

        for key, value in data_tokenizer.word_counts.items():
            tot_cnt = tot_cnt + 1
            tot_freq = tot_freq + value
            if value < thresh:
                cnt = cnt + 1
                freq = freq + value

        # print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
        # print("Total Coverage of rare words:",(freq/tot_freq)*100)
        return tot_cnt, cnt

    def text2seq(self, data, tr_data, tot_cnt, cnt, data_type='p'):
        # prepare a tokenizer for reviews on training data
        data_tokenizer = Tokenizer(num_words=tot_cnt - cnt)
        data_tokenizer.fit_on_texts(list(tr_data))

        # convert text sequences into integer sequences
        data_seq = data_tokenizer.texts_to_sequences(data)
        if data_type == 'x':
            self.reverse_source_word_index = data_tokenizer.index_word
        if data_type == 'y':
            self.reverse_target_word_index = data_tokenizer.index_word
            # self.target_word_index = data_tokenizer.word_index      if end start tokens are added uncomment it.

        return data_seq

    def pad_seq(self, data, maxlenght, padding='post'):
        return pad_sequences(data, maxlen=maxlenght, padding=padding)

    # def seq2summary(self,input_seq):      if end start tokens are added uncomment it.
    #     newString = ''
    #     for i in input_seq:
    #         if ((i != 0 and i != self.target_word_index['sostok']) and i != self.target_word_index['eostok']):
    #             newString = newString + reverse_target_word_index[i] + ' '
    #     return newString

    def seq2summary(self, input_seq):
        try:
            newString = ''
            for i in input_seq:
                if i != 0:
                    newString = newString + self.reverse_target_word_index[i] + ' '
            return newString
        except IndexError:
            loaded_data = self.load_pickle('tokenizerData')
            self.reverse_target_word_index, self.reverse_source_word_index = loaded_data[0], loaded_data[1]
            newString = ''
            for i in input_seq:
                if i != 0:
                    newString = newString + self.reverse_target_word_index[i] + ' '
            return newString

    def seq2text(self, input_seq):
        try:
            newString = ''
            for i in input_seq:
                if i != 0:
                    newString = newString + self.reverse_source_word_index[i] + ' '
            return newString
        except IndexError:
            loaded_data = self.load_pickle('tokenizerData')
            self.reverse_target_word_index, self.reverse_source_word_index = loaded_data[0], loaded_data[1]
            newString = ''
            for i in input_seq:
                if i != 0:
                    newString = newString + self.reverse_source_word_index[i] + ' '
            return newString

    def pickle_data(self, data, filename, path=''):
        pickle_out = open(path + filename + '.pickle', "wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()

    def load_pickle(self, filename, path=''):
        pickle_in = open(path + filename + '.pickle', "rb")
        return pickle.load(pickle_in)
