import os,sys
from utils import parse_excel_data, write_data, parse_ourData_newformat, generate_data_templates

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
print("================\n"+os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import project_statics

# ATIS_DATA_BASE_URL = (
#     "https://github.com/ogrisel/slot_filling_and_intent_detection_of_SLU/blob/"
#     "master/data/atis-2/"
# )

DATA_BASE_URL = (
    os.path.abspath(os.getcwd()) + '\\dataset'
)

ATIS_DATA_BASE_URL = (
    os.path.abspath(os.getcwd()) + '\\dataset\\ATIS\\'
)
# SNIPS_DATA_BASE_URL = (
#     "https://github.com/ogrisel/slot_filling_and_intent_detection_of_SLU/blob/"
#     "master/data/snips/"

# MASSIVE_DATA_BASE_URL = (
#   "https://amazon-massive-nlu-dataset.s3.amazonaws.com/"
# )

# raw file path, save destination path
data = parse_excel_data(DATA_BASE_URL, project_statics.SFID_pickle_files)
temp_train, temp_test = generate_data_templates(DATA_BASE_URL, 3, project_statics.SFID_pickle_files)

data = write_data(data,temp_train, temp_test, ATIS_DATA_BASE_URL)
parse_ourData_newformat(ATIS_DATA_BASE_URL, project_statics.SFID_pickle_files)