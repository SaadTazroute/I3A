# encoding: utf-8

import logging
import torch
import re
def format_to_wa(cfg, model):

    alignment_start_pattern = re.compile(r"<alignment>")
    alignment_end_pattern = re.compile(r"</alignment>")
    equality_pattern = re.compile(r"<==>")
    doubleslash_pattern = re.compile(r"//")

    logger = logging.getLogger("model.wa")
    logger.info("Start inferencing and creating .wa files")
    
    aligment_section = False
    class_dict = {0: 'EQUI', 1 : 'OPPO', 2 : 'REL', 3 : 'SIMI', 4 : 'SPE1',5 : 'SPE2'}
    processed_file = []
    processed_file.append('<sentence id="1" status="">\n')
    file_name = cfg.OUTPUT_DIR + "/" + cfg.DATASETS.TEST_WA[14:-4] + "_predicted.wa"
    with open(cfg.DATASETS.TEST_WA, 'rb') as fin:
        fin.readline()
        for index, line in enumerate(fin):
            processed_line = []
            line = line.decode('latin-1')
            if alignment_end_pattern.match(line):
                aligment_section = False
            if aligment_section:
                equality_pos = [m.start() for m in re.finditer(equality_pattern, line)]
                doubleslash_pos = [m.start() for m in re.finditer(doubleslash_pattern, line)]
                
                alignment = line[:doubleslash_pos[0]]
                value = line[doubleslash_pos[1] + 2 : doubleslash_pos[2]]
                explanation = line[doubleslash_pos[0] + 2 : doubleslash_pos[1]]
                first_chunk = line[doubleslash_pos[2] + 2 : equality_pos[1]]
                second_chunk = line[equality_pos[1] + 4 :]
                
                if value != " NIL " and explanation != ' ALIC ' and explanation != ' NOALI ':
                    
                    tokens = model.roberta.encode(first_chunk, second_chunk)
                    out1, out2 = model(tokens)
                    value = " " + str(round(out1.item())) + " "
                    if explanation.strip() == "EQUI":
                        value = " 5 "
                    _, exp = torch.max(out2, 1)
                    explanation = " " + class_dict[exp.item()] + " "
                
                processed_line = alignment + "//" + explanation + "//" + value + "//" + first_chunk + "<==>" + second_chunk
                processed_file.append(processed_line)
            else:
                processed_file.append(line)

            if alignment_start_pattern.match(line):
                aligment_section = True



   
    with open(file_name, "w", newline='', encoding="utf-8") as f:
        for item in processed_file:
            f.write("%s" % item)

    logger.info('Created file %s' % (file_name))
