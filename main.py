from src.data.data_preprocessing import pre_process_data
from src.utils.utils import parse_arguments

def main():
    # pre process data 

    # take this from an argument and replace hte magic string
    print("1. Main function")
    pre_process_data('bert')

        

if __name__ == "__main__":
    main()