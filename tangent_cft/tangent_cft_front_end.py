import argparse
import logging
import os

from .Embedding_Preprocessing.encoder_tuple_level import TupleTokenizationMode
from .configuration import Configuration
from .wiki_data_reader import WikiDataReader

from .tangent_cft_back_end import TangentCFTBackEnd
from .tangent_cft_module import TangentCFTModule
from .tangent_cft_encoder import FormulaTreeEncoder

import tangent_cft.logging_config as logging_config


def main():
    parser = argparse.ArgumentParser(description='Given the configuration file for training Tangent_CFT model.'
                                                 'This function train the model and then does the retrieval task on'
                                                 'NTCIR-12 formula retrieval task.')

    parser.add_argument('-vocab', type=str, required=True, help="Vocabulary filepath.")
    parser.add_argument('-ftp', type=str, required=True,
                        help="Fasttext model file path (to pretrained or where to save after training.")

    parser.add_argument('--slt', type=bool, action=argparse.BooleanOptionalAction,
                        help="Determines to use slt (True) or opt(False)", default=True)

    parser.add_argument('--et', choices=range(1, 5), default=3, type=int,
                        help='Embedding type; 1:Value, 2:Type, '
                             '3:Type and Value separated and 4: Type and Value Not Separated')
    parser.add_argument('--ifrp', type=bool, action=argparse.BooleanOptionalAction, default=True,
                        help="Determines to ignore full relative path or not")
    parser.add_argument('--ta', type=bool, action=argparse.BooleanOptionalAction, default=False,
                        help="Determines to tokenize all")
    parser.add_argument('--tn', type=bool, action=argparse.BooleanOptionalAction, default=True,
                        help="Determines to tokenize numbers")

    parser.add_argument('--t', type=bool, action=argparse.BooleanOptionalAction, default=False,
                        help="Value True for training a new model and False for loading pretrained model")
    parser.add_argument('--cid', metavar='cid', type=int, help='Fasttext config file id.')
    parser.add_argument('--td', type=str, default=None,
                        help="File path of training data. If using NTCIR12 dataset, "
                             "it should be MathTagArticles directory. "
                             "If using the MSE dataset, it should be csv file of formula")
    parser.add_argument('--etd', type=str, default=None,
                        help="Path to encoded dataset in the format of json "
                             "{document_name: [encoded formulas] or where it will be stored")

    parser.add_argument('--r', type=bool, action=argparse.BooleanOptionalAction, default=False,
                        help="Value True to do the retrieval on NTCIR12 dataset")
    parser.add_argument('--qd', type=str, help="NTCIR12 query directory.", default='./TestQueries')
    parser.add_argument('--rf', type=str, help="Retrieval result file path.", default="ret_res.csv")
    parser.add_argument('--ri', type=int, help="Run Id for Retrieval.", default=1)

    args = vars(parser.parse_args())

    # model params
    vocab_filepath = args['vocab']
    ft_model_filepath = args['ftp']

    # parsing and encoding params
    read_slt = args['slt']
    embedding_type = TupleTokenizationMode(args['et'])
    ignore_full_relative_path = args['ifrp']
    tokenize_all = args['ta']
    tokenize_number = args['tn']

    # training params
    do_train = args['t']
    config_id = args['cid']
    train_data_path = args['td']
    encoded_train_data_path = args['etd']
    if do_train and not config_id and not (train_data_path or encoded_train_data_path):
        raise ValueError('For training mode config_id and either train_data_path or encoded_train_data_path must be '
                         'provided')

    # retrieval params
    do_retrieval = args['r']
    query_dir = args['qd']
    res_file = args['rf']
    run_id = args['ri']
    if do_retrieval and not (train_data_path or encoded_train_data_path):
        raise ValueError('For retrieval you have to provide dataset in train_data_path or encoded_train_data_path')

    data_reader = WikiDataReader(read_slt=read_slt)

    if os.path.isfile(vocab_filepath):
        encoder = FormulaTreeEncoder.load(vocab_filepath)
    else:
        encoder = FormulaTreeEncoder(
            embedding_type=embedding_type,
            ignore_full_relative_path=ignore_full_relative_path,
            tokenize_all=tokenize_all,
            tokenize_number=tokenize_number
        )

    module = TangentCFTModule(ft_model_filepath)
    system = TangentCFTBackEnd(tangent_cft_module=module, encoder=encoder)

    if os.path.isfile(encoded_train_data_path):
        encoded_train_data = system.load_encoded_formulas(encoded_train_data_path)
    else:
        train_data = data_reader.get_collection(train_data_path)
        encoded_train_data = dict(zip(train_data.keys(), system.encoder.fit_transform(list(train_data.values()))))
        system.encoder.save_vocabulary(vocab_filepath)
        system.save_encoded_formulas(encoded_train_data, encoded_train_data_path)

    if do_train:
        logging.info('Start training.')
        config_file_path = "./config/config_" + str(config_id)
        ft_config = Configuration(config_file_path)
        system.train_model(ft_config, list(encoded_train_data.values()), encoded=True)
        system.save_model(ft_model_path=ft_model_filepath, vocabulary_map_path=vocab_filepath)

    if do_retrieval:
        embeddings, formula_ids = system.module.index_collection(encoded_train_data)
        query_encoded = data_reader.get_query(query_dir)
        retrieval_result = system.retrieval(embeddings,
                                            formula_ids,
                                            query_encoded)
        system.create_result_file(retrieval_result, "./Retrieval_Results/" + res_file, run_id)


if __name__ == "__main__":
    logging_config.configure_logging()
    main()
