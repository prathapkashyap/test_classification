from src.data.data_preprocessing import pre_process_data
from src.utils.utils import parse_arguments

def main():
    # pre process data 

    # take this from an argument and replace hte magic string
    print("1. Main function")
    args = parse_arguments()
    passed_model = args.model
    # logger = init_log()
    # logger.info("parsing arguments", passed_model)
    pre_process_data(passed_model)
    # logger = init_log()

        

if __name__ == "__main__":
    main()