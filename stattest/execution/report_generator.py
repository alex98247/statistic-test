import math

import numpy as np
from fpdf import FPDF

from stattest.execution.cache import CacheResultService


class AbstractReportBlockGenerator:
    def build(self, pdf):
        raise NotImplementedError("Method is not implemented")


class ImageReportBlockGenerator(AbstractReportBlockGenerator):
    def build(self, pdf):
        raise NotImplementedError("Method is not implemented")

    @staticmethod
    def add_image(pdf, image, h, w, x, y):
        pdf.image(image, h=h, w=w, x=x, y=y)


class TableReportBlockGenerator(AbstractReportBlockGenerator):
    def build(self, pdf):
        raise NotImplementedError("Method is not implemented")

    @staticmethod
    def add_table(pdf, header: [tuple], data: [tuple], col_width=None, line_height=None, border=1,
                  max_line_height=None):
        result = [[str(x) for x in tup] for tup in header] + [[str(x) for x in tup] for tup in data]

        if col_width is None:
            col_width = pdf.epw / len(header[0])

        if line_height is None:
            line_height = pdf.font_size * 2.5

        if max_line_height is None:
            max_line_height = pdf.font_size

        for row in result:
            for datum in row:
                pdf.multi_cell(col_width, line_height, datum, border=border, ln=3,
                               max_line_height=max_line_height)
            pdf.ln(line_height)

        return line_height * len(result)


class ReportGenerator:

    def __init__(self, generators: [AbstractReportBlockGenerator], font='Times', padding=5):
        self.pdf = FPDF(orientation='L')
        self.pdf.set_font(font)
        self.pdf.add_page()
        self.generators = generators
        self.padding = padding

    def generate(self, path='report.pdf'):
        start_y = self.pdf.get_y()
        for generator in self.generators:
            height = generator.build(self.pdf)
            start_y = start_y + height + self.padding
            self.pdf.set_y(start_y)

        self.pdf.output(path)


class PowerTableReportBlockGenerator(TableReportBlockGenerator):
    def __init__(self, data_path='result/result.json'):
        self.cache = CacheResultService(filename=data_path, separator=':')

    def build_table_alternative(self, pdf, alternative):
        significant_levels = self.cache.get_level_prefixes([alternative], 1)
        height = 0
        for significant_level in significant_levels:
            height = height + self.build_table(pdf, alternative, significant_level)

        return height

    def build_table(self, pdf, alternative, significant_level):
        pdf.text(y=pdf.get_y() + 10, x=10,
                 text='Alternative: ' + alternative + ' Significant level: ' + significant_level)
        pdf.set_y(pdf.get_y() + 15)

        values = self.cache.get_with_prefix([alternative, significant_level])

        test_codes_all = set()
        sizes = set()
        for key in values:
            split = key.split(':')
            test_codes_all.add(split[-1])
            sizes.add(int(split[-2]))

        test_codes_all = list(test_codes_all)
        sizes = list(sizes)
        sizes.sort()
        data = [[0] * (len(sizes) + 1) for i in range(len(test_codes_all))]
        for i in range(len(test_codes_all)):
            data[i][0] = test_codes_all[i]
        for key in values:
            split = key.split(':')
            index_test = test_codes_all.index(split[-1])
            index_size = sizes.index(int(split[-2]))
            data[index_test][index_size + 1] = round(values[key], 2)

        return self.add_table(pdf, [['Test'] + sizes], data) + 10

    def build(self, pdf):
        alternatives = self.cache.get_level_prefixes([], 0)
        height = 0
        for alternative in alternatives:
            height = height + self.build_table_alternative(pdf, alternative)

        return height


class TopPowerTableReportBlockGenerator(TableReportBlockGenerator):
    def __init__(self, data_path='result/result.json'):
        self.cache = CacheResultService(filename=data_path, separator=':')

    def build_table_alternative(self, pdf, alternative):
        significant_levels = self.cache.get_level_prefixes([alternative], 1)
        height = 0
        for significant_level in significant_levels:
            height = height + self.build_table(pdf, alternative, significant_level)

        return height

    def build_table(self, pdf, alternative, significant_level):
        pdf.text(y=pdf.get_y() + 10, x=10,
                 text='Alternative: ' + alternative + ' Significant level: ' + significant_level)
        pdf.set_y(pdf.get_y() + 15)

        values = self.cache.get_with_prefix([alternative, significant_level])

        test_codes_all = set()
        sizes = set()
        for key in values:
            split = key.split(':')
            test_codes_all.add(split[-1])
            sizes.add(int(split[-2]))

        test_codes_all = list(test_codes_all)
        sizes = list(sizes)
        sizes.sort()
        data = [[0] * (len(sizes) + 1) for i in range(len(test_codes_all))]
        for i in range(len(test_codes_all)):
            data[i][0] = test_codes_all[i]
        for key in values:
            split = key.split(':')
            index_test = test_codes_all.index(split[-1])
            index_size = sizes.index(int(split[-2]))
            data[index_test][index_size + 1] = round(values[key], 5)

        for i in range(len(sizes)):
            ind = np.argpartition(data[:][i], -3)[-3:]


        return self.add_table(pdf, [sizes], data) + 10

    def build(self, pdf):
        alternatives = self.cache.get_level_prefixes([], 0)
        height = 0
        for alternative in alternatives:
            height = height + self.build_table_alternative(pdf, alternative)

        return height


class PowerTableReportTimeBlockGenerator(TableReportBlockGenerator):
    def __init__(self, data_path='times.json'):
        self.cache = CacheResultService(filename=data_path, separator=':')

    def build(self, pdf):
        values = self.cache.get_with_prefix([])

        test_codes_all = set()
        sizes = set()
        for key in values:
            split = key.split(':')
            test_codes_all.add(split[1])
            sizes.add(int(split[0]))

        test_codes_all = list(test_codes_all)
        sizes = list(sizes)
        sizes.sort()
        data = [[0] * (len(sizes) + 1) for i in range(len(test_codes_all))]
        for i in range(len(test_codes_all)):
            data[i][0] = test_codes_all[i]
        for test in test_codes_all:
            for size in sizes:
                index_test = test_codes_all.index(test)
                index_size = sizes.index(size)
                mean = self.cache.get_with_level([str(size), test, 'mean'])
                std = self.cache.get_with_level([str(size), test, 'std'])
                data[index_test][index_size + 1] = str(round(mean, 4)) + ', ' + str(round(std, 4))

        return self.add_table(pdf, [['Test'] + sizes], data) + 10


"""
if __name__ == '__main__':
    report_generator = ReportGenerator([PowerTableReportTimeBlockGenerator()])
    report_generator.generate()
"""
