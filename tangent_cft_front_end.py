import argparse
import logging

from Embedding_Preprocessing.encoder_tuple_level import TupleTokenizationMode
from Configuration.configuration import Configuration
from DataReader.mse_data_reader import MSEDataReader
from DataReader.wiki_data_reader import WikiDataReader
from tangent_cft_back_end import TangentCFTBackEnd
import logging_config


def main():
    parser = argparse.ArgumentParser(description='Given the configuration file for training Tangent_CFT model.'
                                                 'This function train the model and then does the retrieval task on'
                                                 'NTCIR-12 formula retrieval task.')

    parser.add_argument('--t', type=bool, action=argparse.BooleanOptionalAction,
                        help="Value True for training a new model and False for loading a model", default=True)
    parser.add_argument('--r', type=bool, help="Value True to do the retrieval on NTCIR12 dataset",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-ds', type=str,
                        help="File path of training data. If using NTCIR12 dataset, "
                             "it should be MathTagArticles directory. If using the MSE dataset, it"
                             "should be csv file of formula", required=True)
    parser.add_argument('-cid', metavar='cid', type=int, help='Configuration file.', required=True)
    parser.add_argument('--wiki', type=bool, action=argparse.BooleanOptionalAction,
                        help="Determines if the dataset is wiki or not.", default=True)
    parser.add_argument('--slt', type=bool, action=argparse.BooleanOptionalAction,
                        help="Determines to use slt (True) or opt(False)", default=True)
    parser.add_argument('-em', type=str, help="File path for encoder map.", required=True)
    parser.add_argument('--mp', type=str, help="Model file path.", default=None)
    parser.add_argument('--qd', type=str, help="NTCIR12 query directory.", default=None)
    parser.add_argument('--rf', type=str, help="Retrieval result file path.", default="ret_res")
    parser.add_argument('--ri', type=int, help="Run Id for Retrieval.", default=1)
    parser.add_argument('--frp', type=bool, action=argparse.BooleanOptionalAction,
                        help="Determines to ignore full relative path", default=True)
    parser.add_argument('--ta', type=bool, action=argparse.BooleanOptionalAction,
                        help="Determines to tokenize all", default=False)
    parser.add_argument('--tn', type=bool, action=argparse.BooleanOptionalAction,
                        help="Determines to tokenize numbers", default=True)
    parser.add_argument('--et', help='Embedding type; 1:Value, 2:Type, 3:Type and Value separated and'
                                     ' 4: Type and Value Not Separated', choices=range(1, 5),
                        default=3, type=int)
    parser.add_argument('--eds', type=str, help="Path to encoded dataset in the format of json "
                                                "{document_name: [encoded formulas] or where it will be stored",
                        default=None)

    args = vars(parser.parse_args())

    train_model = args['t']
    do_retrieval = args['r']
    dataset_file_path = args['ds']
    config_id = args['cid']
    is_wiki = args['wiki']
    read_slt = args['slt']
    encoder_file_path = args['em']
    model_file_path = args['mp']
    res_file = args['rf']
    run_id = args['ri']
    ignore_full_relative_path = args['frp']
    tokenize_all = args['ta']
    tokenize_number = args['tn']
    queries_directory_path = args['qd']
    embedding_type = TupleTokenizationMode(args['et'])
    encoded_dataset = args['eds']

    map_file_path = "Embedding_Preprocessing/" + str(encoder_file_path)
    config_file_path = "Configuration/config/config_" + str(config_id)
    config = Configuration(config_file_path)

    if is_wiki:
        data_reader = WikiDataReader(dataset_file_path, read_slt=read_slt,
                                     queries_directory_path=queries_directory_path)
    else:
        data_reader = MSEDataReader(dataset_file_path, read_slt=read_slt)

    system = TangentCFTBackEnd(config=config, data_reader=data_reader)

    if train_model:
        logging.info("Training Tangent_CFT model with the following parameters:"
                     f"map_file_path: {map_file_path},"
                     f"model_path: {model_file_path},"
                     f"embedding_type: {embedding_type},"
                     f"ignore_full_relative_path: {ignore_full_relative_path},"
                     f"tokenize_all: {tokenize_all},"
                     f"tokenize_number: {tokenize_number}"
                     )

        embeddings, formula_ids = system.train_model(
            encoder_map_path=map_file_path,
            ft_model_path=model_file_path,
            encoded_train_formulas=encoded_dataset,
            embedding_type=embedding_type,
            ignore_full_relative_path=ignore_full_relative_path,
            tokenize_all=tokenize_all,
            tokenize_number=tokenize_number
        )

        if do_retrieval:
            retrieval_result = system.retrieval(embeddings,
                                                formula_ids,
                                                embedding_type,
                                                ignore_full_relative_path,
                                                tokenize_all,
                                                tokenize_number
                                                )
            system.create_result_file(retrieval_result, "Retrieval_Results/" + res_file, run_id)
    else:

        logging.info(f"Loading model with the following parameters: "
                     f"map_file_path: {map_file_path},"
                     f"model_path: {model_file_path},"
                     f"embedding_type: {embedding_type},"
                     f"ignore_full_relative_path: {ignore_full_relative_path},"
                     f"tokenize_all: {tokenize_all},"
                     f"tokenize_number: {tokenize_number}")

        system.load_model(
            encoder_map_path=map_file_path,
            ft_model_path=model_file_path
        )

        encoded_formulas = system.load_encoded_formulas(encoded_dataset)
        train_embeddings, formula_ids = system.module.index_collection(encoded_formulas)

        if do_retrieval:
            retrieval_result = system.retrieval(train_embeddings,
                                                formula_ids,
                                                embedding_type,
                                                ignore_full_relative_path,
                                                tokenize_all,
                                                tokenize_number
                                                )
            system.create_result_file(retrieval_result, "Retrieval_Results/" + res_file, run_id)


if __name__ == "__main__":
    logging_config.configure_logging()
    main()
