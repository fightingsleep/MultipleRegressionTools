import csv
import argparse
import sys








def malformed(row):
    if ("NA" in row):
        return True
    return False


def read_csv(filename):
    csvfile = open(filename)
    csvreader = csv.reader(csvfile)
    line_count = 1
    headers = []
    data = []
    for row in csvreader:
        if (line_count == 1):
            headers = row
        if (not malformed(row)):
            data.append(row)
        line_count += 1
    return {'headers': headers, 'data': data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='<input file path>',
                        help='input file containing the data for linear regression')
    parser.print_help()
    args = parser.parse_args()
    data = read_csv(args.input)
    print(data)

if __name__ == "__main__":
    main()
