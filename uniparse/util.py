"""Utility functions."""
import re
import os.path
import functools


def validate_line(line):
    """Validate UD line."""
    if line.startswith("#"):
        return False
    if line == "\n":
        return True
    # think this is the one we want
    if not re.match(r"\d+\t", line):
        return False

    return True


UD_DEFAULT_LINE = [
    "1",  # index
    "_",  # form
    "_",  # lemma
    "_",  # upos
    "_",  # xpos
    "_",  # feats
    "_",  # head
    "_",  # deprel
    "_",  # deps
    "_",  # misc
]


def convert_delimited_to_conllu(filename, delimiter="\t", *, output_filename=None):
    """
    Convert a delimited file and converts it into UD.

    This converter accepts between 1-3 delimited values, each with their mening:
        1. Form
        2. Tag
    """

    filehandler_in = open(filename, encoding="UTF-8")
    if not output_filename:
        output_filename = os.path.splitext(filename)[0] + ".conllu"

    filehandler_out = open(output_filename, "w", encoding="UTF-8")

    index = 1
    for line in filehandler_in:
        if line == "\n":
            line = ""
            index = 1
        else:
            columns = line.strip().split(delimiter)
            line_list = UD_DEFAULT_LINE.copy()
            line_list[0] = str(index)
            line_list[1] = columns[0]
            line_list[3] = columns[1] if len(columns) > 1 else line_list[3]
            line = "\t".join(line_list)
            index += 1

        print(line, file=filehandler_out)  # print empty line

    print(file=filehandler_out)

    filehandler_in.close()
    filehandler_out.close()

    return output_filename


def write_predictions_to_file(predictions, reference_file, output_file, vocab):
    if not predictions:
        raise ValueError("No predictions to write to file.")
    indices, arcs, rels = zip(*predictions)
    flat_arcs = _flat_map(arcs)
    flat_rels = _flat_map(rels)

    idx = 0
    with open(reference_file, encoding="UTF-8") as f, open(output_file, 'w', encoding="UTF-8") as fo:
        for line in f.readlines():
            if re.match(r'\d+\t', line):
                info = line.strip().split()
                assert len(info) == 10, 'Illegal line: %s' % line
                info[6] = str(flat_arcs[idx])
                info[7] = vocab.id2rel(flat_rels[idx])
                fo.write('\t'.join(info) + '\n')
                idx += 1
            else:
                fo.write(line)

def _flat_map(lst):
    return functools.reduce(lambda x, y: x + y, [list(result) for result in lst])
