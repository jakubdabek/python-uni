import fileinput
import os
import sys

if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print(f"usage: {sys.argv[0]} <filename>")
    #     sys.exit(1)

    byte_count = 0
    line_count = 0
    word_count = 0
    max_line_length = -1
    with fileinput.input(mode='rb') as f:
        for line in f:
            line: bytes
            byte_count += len(line)
            line_count += 1
            word_count += len(line.split())
            without_endline = line.rstrip(b'\r\n')
            max_line_length = max(len(without_endline), max_line_length)

    print("byte count: ", byte_count)
    print("line count: ", line_count)
    print("word count: ", word_count)
    print("maximal line length: ", max_line_length)
