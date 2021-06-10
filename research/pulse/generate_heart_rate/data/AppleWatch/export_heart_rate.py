from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from xml.etree import ElementTree
from collections import Counter, OrderedDict
import numpy as np
import csv
import datetime
import re
import sys
import os
os.chdir(os.path.dirname(__file__))


WORK_DIR = 'Series_5/KeDei/'  # 作業ディレクトリ
DATA_FILE_NAME = 'export.xml'  # データファイル名
SEPARATE_FILE_NAME = 'run.log'  # 分割ファイル名
EXPORT_FILE_NAME = 'heart_rate.csv'  # 出力ファイル名

# 使用データ
RECORD_FIELDS = [
    'device',
    'startDate',
    'value'
]


PREFIX_RE = re.compile('^HK.*TypeIdentifier(.+)$')
ABBREVIATE = True
VERBOSE = True


def format_freqs(counter):
    """
    Format a counter object for display.
    """
    return '\n'.join('%s: %d' % (tag, counter[tag])
                     for tag in sorted(counter.keys()))


def format_value(value, field):
    """
    Format a value for a CSV file, escaping double quotes and backslashes.
    """
    if field == 'device':
        for split_value in value.split():
            if 'hardware' in split_value:
                for hardware in split_value.split(','):
                    if 'hardware' in hardware:
                        return hardware.split(':')[1]
    elif field == 'startDate':
        return value.split('+')[0].rstrip()
    elif field == 'value':
        return value
    else:
        raise KeyError('Unexpected format value: %s' % datatype)


def abbreviate(s, enabled=ABBREVIATE):
    """
    Abbreviate particularly verbose strings based on a regular expression
    """
    m = re.match(PREFIX_RE, s)
    return m.group(1) if enabled and m else s


def encode(s):
    """
    Encode string for writing to file.
    In Python 2, this encodes as UTF-8, whereas in Python 3,
    it does nothing
    """
    return s.encode('UTF-8') if sys.version_info.major < 3 else s


class HealthDataExtractor(object):
    """
    Extract health data from Apple Health App's XML export, export.xml.

    Inputs:
        path:      Relative or absolute path to export.xml
        verbose:   Set to False for less verbose output

    Outputs:
        Writes a CSV file for each record type found, in the same
        directory as the input export.xml. Reports each file written
        unless verbose has been set to False.
    """

    def __init__(self, path, verbose=VERBOSE):
        self.in_path = path
        self.verbose = verbose
        self.directory = os.path.abspath(os.path.split(path)[0])
        with open(path, encoding='UTF-8') as f:
            self.report('Reading data from %s . . . ' % path, end='')
            self.data = ElementTree.parse(f)
            self.report('done')
        self.root = self.data._root
        self.nodes = list(self.root)
        self.n_nodes = len(self.nodes)
        self.abbreviate_types()
        self.collect_stats()

    def report(self, msg, end='\n'):
        if self.verbose:
            print(msg, end=end)
            sys.stdout.flush()

    def count_tags_and_fields(self):
        self.tags = Counter()
        self.fields = Counter()
        for record in self.nodes:
            self.tags[record.tag] += 1
            for k in record.keys():
                self.fields[k] += 1

    def count_record_types(self):
        """
        Counts occurrences of each type of (conceptual) "record" in the data.

        In the case of nodes of type 'Record', this counts the number of
        occurrences of each 'type' or record in self.record_types.

        The slightly different handling reflects the fact that 'Record'
        nodes come in a variety of different subtypes that we want to write
        to different data files.
        """
        self.record_types = Counter()
        self.other_types = Counter()
        for record in self.nodes:
            if record.tag == 'Record' and record.attrib['type'] == 'HeartRate':
                self.record_types[record.attrib['type']] += 1

    def collect_stats(self):
        self.count_record_types()
        self.count_tags_and_fields()

    def open_for_writing(self):
        self.handles = {}
        self.paths = []
        for kind in (list(self.record_types) + list(self.other_types)):
            path = os.path.join(self.directory, EXPORT_FILE_NAME)
            f = open(path, 'w', encoding='UTF-8')
            f.write(','.join(RECORD_FIELDS) + '\n')
            self.handles[kind] = f
            self.report('Opening %s for writing' % path)

    def abbreviate_types(self):
        """
        Shorten types by removing common boilerplate text.
        """
        for node in self.nodes:
            if node.tag == 'Record':
                if 'type' in node.attrib:
                    node.attrib['type'] = abbreviate(node.attrib['type'])

    def write_records(self):
        for node in self.nodes:
            if node.tag == 'Record' and node.attrib['type'] == 'HeartRate':
                attributes = node.attrib
                kind = attributes['type']
                values = [format_value(attributes.get(field), field)
                          for field in RECORD_FIELDS]
                line = encode(','.join(values) + '\n')
                self.handles[kind].write(line)

    def close_files(self):
        for (kind, f) in self.handles.items():
            f.close()
            self.report('Written %s data.' % abbreviate(kind))

    def extract(self):
        self.open_for_writing()
        self.write_records()
        self.close_files()

    def report_stats(self):
        print('\nTags:\n%s\n' % format_freqs(self.tags))
        print('Fields:\n%s\n' % format_freqs(self.fields))
        print('Record types:\n%s\n' % format_freqs(self.record_types))


def separate_files(filedir, filename, separate):
    """
    ファイルの分割

    Args:
        filedir (string): 分割するファイルのディレクトリ
        filename (string): ファイル名
        separate (string): 分割用ファイル名
    """

    def str2datetime(date_time, date_format):
        """
        日時の変換（文字列 -> オブジェクト）

        Args:
            date_time (string): 変換する文字列
            date_format (string): 切り出し文字列
        Returns:
            :obj:`datetime`: 日時オブジェクト
        """

        temp = datetime.datetime.strptime(date_time, date_format)
        return datetime.datetime(temp.year, temp.month, temp.day, temp.hour, temp.minute, temp.second)

    # データの読み出し
    data = []
    with open(filedir + filename, encoding='UTF-8') as f:
        reader = csv.reader(f)

        # ヘッダーのスキップ
        next(reader)

        for row in reader:
            # データの追加
            data.append(row)
    print('\nRead Data: ' + str(len(data)))

    # 日時でソート
    sorted_data = sorted(
        data, key=lambda x: str2datetime(x[1], '%Y-%m-%d %H:%M:%S'))

    # 分割箇所データの作成
    separate_data = []
    with open(filedir + separate, encoding='UTF-8') as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) == 3:
                # データの追加
                separate_data.append(
                    [str2datetime(row[0], '%Y/%m/%d %H:%M:%S'), row[2]])

    write_num = 0
    data = []
    separate_index = 1
    for row in sorted_data:
        date_time = str2datetime(row[1], '%Y-%m-%d %H:%M:%S')
        # 分割時刻になったら書き出し
        if separate_index < len(separate_data) and date_time > separate_data[separate_index][0]:
            # ファイルに書き出して分割
            with open(WORK_DIR + separate_data[separate_index-1][0].strftime('%Y%m%d_%H%M%S_') + separate_data[separate_index-1][1] + '.csv', 'w', newline='') as export_file:
                export_writer = csv.writer(export_file, delimiter=',')
                export_writer.writerows(data)
                write_num += len(data)
                data = []
                separate_index += 1

        # データの追加
        data.append(row)

    # 最後のデータの書き出し
    with open(WORK_DIR + separate_data[separate_index-1][0].strftime('%Y%m%d_%H%M%S_') + separate_data[separate_index-1][1] + '.csv', 'w', newline='') as export_file:
        export_writer = csv.writer(export_file, delimiter=',')
        export_writer.writerows(data)
        write_num += len(data)

    print('Write Data: ' + str(write_num))


def main():
    # ファイルの書き出し
    if os.path.isfile(WORK_DIR + EXPORT_FILE_NAME) is False:
        data = HealthDataExtractor(WORK_DIR + DATA_FILE_NAME)
        data.report_stats()
        data.extract()

    # ファイルの分割
    separate_files(WORK_DIR, EXPORT_FILE_NAME, SEPARATE_FILE_NAME)


if __name__ == '__main__':
    main()
