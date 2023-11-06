
def main():
    with open("shuffled-pl-en") as input_file:
        text_lines = input_file.read().split('\n')

    f = open('medical_corpus.txt', 'w', encoding='utf-8')
    print("Writing medical corpus to file now")

    id = 0

    for text_line in text_lines:
        parts = text_line.strip().split("\t")
        if len(parts) != 4:
            continue
        if parts[2] == "medical_corpus":
            f.write(str(id) + "\t" + text_line)
            f.write('\n')
            id += 1
    
    f.close()

if __name__ == '__main__':
    main()