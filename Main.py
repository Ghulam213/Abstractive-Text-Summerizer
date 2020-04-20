from DataPreprocessing import DataPreprocessing


class Main():
    def data_processing(self):
        processor = DataPreprocessing()
        # load Dataset
        df = processor.load_dataset('cnn')
        print('loaded')
        # clean Dataset
        for index, example in df.iterrows():
            example['Stories'] = processor.clean_stories(example['Stories'].split('\n'))
            example['Highlights'] = processor.clean_highlights(example['Highlights'])

        df = processor.drop_dulp_and_na(df, ["Stories", "Highlights"])
        df = processor.clear_long_text(df)    # drop stories longer than 700 words and summaries longer than 38 words
        print('Cleaned')
        #uncomment this if start and end tokens are to be added.
        #the current data is processed without it.
        df['Highlights'] = processor.start_end_token(df['Highlights'])
        print('end tokened')

        x = df["Stories"]
        y = df["Highlights"]

        # Converting textual data into int based seq using keras tokenizer class
        # the rare word count gives all rare words which are used less than 5 times in whole dataset
        # it is need to make proper sequences

        # for x(which are stories in df)
        total_word, rare_word = processor.rare_words_count(x)
        x_seq = processor.text2seq(x, x, total_word, rare_word, data_type='x')

        # for y(which are highlights in dataset)
        total_word, rare_word = processor.rare_words_count(y)
        y_seq = processor.text2seq(y, y, total_word, rare_word, data_type='y')
        print("sequenced")

        # seq are paded to max lenght that is 700 in case of stories by adding 0 at the end
        x_seq = processor.pad_seq(x_seq, processor.max_stories_len)

        # seq are paded to max lenght that is 38 in case of stories by adding 0 at the end
        y_seq = processor.pad_seq(y_seq, processor.max_highlights_len)
        print("paded")

        x_tr, x_test, x_dev, y_tr, y_test, y_dev = processor.split_data(x_seq, y_seq,
                                                                        train_ratio=0.75, dev_ratio=0.125)
        print("splited")

        processor.pickle_data([x_tr, x_test, x_dev, y_tr, y_test, y_dev], 'Data_StrForm')

        # to load it use
        # data =  processor.load_pickle('Data_StrFrom')
        # x_tr, x_test, x_dev, y_tr, y_test, y_dev = data[0],data[1],data[2],data[3],data[4],data[5]

        processor.pickle_data([processor.reverse_target_word_index, processor.reverse_source_word_index],
                              'text2seqData')
        # to load it use
        # loaded_data = self.load_pickle('tokenizerData')
        # processor.reverse_target_word_index, processor.reverse_source_word_index = loaded_data[0], loaded_data[1]
        print("pickled")


if __name__ == "__main__":
    main = Main()
    main.data_processing()
