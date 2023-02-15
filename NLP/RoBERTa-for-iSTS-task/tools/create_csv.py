# encoding: utf-8
import re
import csv
import sys
sys.path.append('.')
from net_config import cfg
from utils.logger import setup_logger

def create_csv(data_files, file_name):

    alignment_start_pattern = re.compile(r"<alignment>")
    alignment_end_pattern = re.compile(r"</alignment>")
    equality_pattern = re.compile(r"<==>")
    doubleslash_pattern = re.compile(r"//")

    aligment_section = False
    processed_files = []

    for file_path in data_files:
        processed_file = []
        with open(file_path, 'rb') as fin:
            fin.readline()
            for index, line in enumerate(fin):
                processed_line = []
                line = line.decode('latin-1')
                if alignment_end_pattern.match(line):
                    aligment_section = False
                if aligment_section:
                    equality_pos = [m.start() for m in re.finditer(equality_pattern, line)]
                    doubleslash_pos = [m.start() for m in re.finditer(doubleslash_pattern, line)]
                    
                    value = line[doubleslash_pos[1] + 2 : doubleslash_pos[2]].strip().strip("\n")
                    explanation = line[doubleslash_pos[0] + 2 : doubleslash_pos[1]].strip().strip("\n")
                    first_chunk = line[doubleslash_pos[2] + 2 : equality_pos[1]].strip().strip("\n")
                    second_chunk = line[equality_pos[1] + 4 :].strip().strip("\n")

                    processed_line.append(first_chunk)
                    processed_line.append(second_chunk)
                    if value != "NIL" and len(explanation) <= 4  and explanation != 'ALIC':
                        processed_line.append(int(value))
                        processed_line.append(explanation)
                        processed_file.append(processed_line)

                if alignment_start_pattern.match(line):
                    aligment_section = True

        processed_files += processed_file

    with open(file_name, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(['chunk1', 'chunk2', 'value', 'explanation'])
        writer.writerows(processed_files)
    

  
def main():
  
    train_files = ['semeval_data/train/STSint.gs.headlines.wa',
                    'semeval_data/train/STSint.gs.images.wa',
                    'semeval_data/train_students_answers_2015_10_27.utf-8/STSint.input.answers-students.wa']
    test_files = ['semeval_data/test_goldStandard/STSint.testinput.answers-students.wa',
                'semeval_data/test_goldStandard/STSint.testinput.headlines.wa',
                'semeval_data/test_evaluation_task2c/STSint.gs.headlines.wa',
                'semeval_data/test_evaluation_task2c/STSint.gs.images.wa']

    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger("model", output_dir, 0)
    
    create_csv(train_files, "data/datasets/train.csv")
    logger.info("Created data/datasets/train.csv")
    create_csv(test_files, "data/datasets/test.csv")
    logger.info("Created data/datasets/test.csv")
    create_csv([test_files[0]], "data/datasets/test_answers-students.csv")
    logger.info("Created data/datasets/test_answers-students.csv")
    create_csv([test_files[2]], "data/datasets/test_headlines.csv")
    logger.info("Created data/datasets/test_headlines.csv")
    create_csv([test_files[3]], "data/datasets/test_images.csv")
    logger.info("Created data/datasets/test_images.csv")


if __name__ == '__main__':
    main()
