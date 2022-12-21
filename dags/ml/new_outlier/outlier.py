from sqlalchemy.engine import Connection

from ml.new_outlier.types.filter_request_dto import FilterRequestDTO
from ml.new_outlier.sql_templates import OutlierSQLTemplate
from ml.new_outlier.utils import random_string, pairs


class OutlierManager:

    def __init__(
            self,
            table_schema: str,
            table_name: str,
            metric_type: str,
            sections: list,
            minimum_rows: int,
            outlier_num: int,
            use_case_branch: bool,
            lower_bound: float,
            upper_bound: float,
            connection: Connection,
    ):
        self.templater = OutlierSQLTemplate(
            table_schema=table_schema,
            table_name=table_name,
            metric_type=metric_type,
            outlier_num=outlier_num,
            min_row_count=minimum_rows,
            use_case_branch=use_case_branch,
            groups=sections,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        self.connection = connection

    @staticmethod
    def get_batch_borders(size):
        delta_batch_size = 2000
        batches = int(size) // delta_batch_size
        batches += 1 if batches % delta_batch_size != 0 else 0
        if batches < 2:
            batches = 2
        batch_borders = list(i / batches for i in range(batches + 1))
        return batch_borders[1:-1]

    def check_size(self):
        query = self.templater.generate_table_size_checking_query()
        size = self.connection.execute(query).fetchall()[0][1]  # [0] -> first row : [1] -> is size in mb
        return size

    def get_case_borders(self, borders: list):
        query = self.templater.generate_batch_case_quantiles_query(borders)
        case_borders = self.connection.execute(query).fetchall()[0]
        return case_borders

    def create_outliers_table(self):
        query = self.templater.generate_resulting_table_creation_query()
        self.connection.execute(query)

    def create_temp_quantiles_table(self, quantile_table_name):
        query = self.templater.generate_quantiles_table_creation_query(quantile_table_name)
        self.connection.execute(query)

    def drop_temp_quantiles_table(self, quantile_table_name):
        query = self.templater.generate_quantiles_table_drop_query(quantile_table_name)
        self.connection.execute(query)

    def get_batch_quantiles(self, borders, quantile_table_name):
        queries = []
        for left_border, right_border in pairs(borders):
            query = self.templater.generate_calculation_outliers_batch_query(left_border, right_border,
                                                                             quantile_table_name)
            yield query

    def run(self):
        quantile_table_name = random_string()
        size = self.check_size()
        borders = self.get_batch_borders(size)
        borders = self.get_case_borders(borders)
        print(borders)
        self.create_outliers_table()
        self.create_temp_quantiles_table(quantile_table_name)
        for query in self.get_batch_quantiles(borders, quantile_table_name):
            print(query)
            self.connection.execute(query)
        self.drop_temp_quantiles_table(quantile_table_name)



