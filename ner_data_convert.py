# -*-coding:utf-8 -*-
import codecs
import os
if __name__ == "__main__":
    data_path = "ner_data"
    infile = codecs.open(os.path.join(data_path, "example.train"), encoding="utf-8")
    chars = []
    tags = []

    for line in infile:
        line_list = line.strip().split(" ")
        if len(line_list) >= 2:
            chars.append(line_list[0])
            tags.append(line_list[1])
        else:
            chars.append("<OOV>")
            tags.append("OOV")

    new_tags = []
    for i, tag in enumerate(tags):
        if tag != "OOV":
            if tag == 'O':
                new_tags.append(tag)
            elif tag.split('-')[0] == 'B':
                if i + 1 != len(tags) and \
                                tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('B-', 'S-'))
            elif tag.split('-')[0] == 'I':
                if i + 1 < len(tags) and \
                                tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('I-', 'E-'))
            else:
                raise Exception('Invalid IOB format!')
        else:
            new_tags.append("<OOV>")
            continue

    outfile = codecs.open(os.path.join(data_path, "train_data.txt"), encoding="utf-8", mode="w")
    for char, tag in zip(chars, new_tags):
        if tag != "<OOV>":
            outfile.write(str(char)+" "+str(tag)+"\n")
        else:
            outfile.write("\n")
