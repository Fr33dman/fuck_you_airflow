from enum import Enum, auto
from os.path import abspath, dirname


TEXT_DATA_PATH = abspath(dirname(__file__))
MISTAKE_LEMMAS_PATH = f"{TEXT_DATA_PATH}/autoinsight_text_data/mistake_words_ru.txt"
REVERSAL_LEMMAS_PATH = f"{TEXT_DATA_PATH}/autoinsight_text_data/reversal_words_ru.txt"


class DataHolderFeatures(str, Enum):
    """Special are in use, general are just for clarification"""

    # general
    id_ = "id_column"  # categorical
    activity = "activity_column"  # categorical
    start_time = "start_timestamp_column"  # numeric
    end_time = "end_timestamp_column"  # numeric
    text = "text_column"  # categorical

    # special
    duration = "duration"
    mistake = "text_mistake"
    reversal = "text_reversal"

    def __str__(self) -> str:
        return self.value


class AnomalyDetectionParameter(Enum):
    dbscan_eps = auto()
    contamination = auto()

    def __str__(self) -> str:
        return self.name


class TextProblemTypes(str, Enum):
    mistake = auto()
    reversal = auto()

    def __str__(self) -> str:  # FIXME DataHolderFeatures[problem_type.name] later
        return self.name


# TODO make dataclass with calculations
class ActivityComputedMetrics(str, Enum):
    mean_time = auto()
    diff_time = auto()
    frequency = auto()
    cycles = auto()
    mean_median = auto()
    duplicates_rate = auto()
    activities_num = auto()
    anomaly_level = auto()

    def __str__(self) -> str:
        return self.name


class ActivityInsightfulMetrics(str, Enum):
    prolonged = "Высокая длительность этапа"
    monotonically_increasing_duration = "Длительность этапа растет со временем"
    irregular = "Нерегулярный этап"
    has_loops = "Зацикленность"
    tight_bottleneck = "Неоптимально организованный редкий этап"
    widespread_bottleneck = "Оптимальный, но неэффективно выполняемый этап"
    numerous_incident_stagnation = "Большое число аномально долгих выполнений этапа"
    distinctive_incidents_affect_duration = "Разовые случаи аномально долгих выполнений этапа"
    duration_increase_of_process_and_or_related_activities = (
        "Рост времени процесса вследствие наличия данного этапа"
    )
    system_errors_stagnation = "Наличие на этапе ошибок, которые не приводят к неуспешному завершению процесса"
    system_errors_fail = (
        "Наличие на этапе системных ошибок (в АС), которые приводят к неуспешному завершению процесса"
    )
    system_design_errors_fail = (
        "Наличие на этапе структурных ошибок, которые приводят к неуспешному завершению процесса"
    )
    reversal_stagnation = (
        "Возвраты и исправления на данном этапе, которые не приводят к неуспешному завершению процесса"
    )
    reversal_fail = "Возвраты и исправления на данном этапе, которые приводят к неуспешному завершению процесса"

    def __str__(self) -> str:
        return self.value


class ActivityInsightsOutcome(str, Enum):
    anomaly_level = "Уровень аномальности"
    fin_effect_amount = "Сумма финансовых эффектов"

    def __str__(self) -> str:
        return self.value


NO_FIN_EFFECT_IMPLEMENTED_METRICS = [
    ActivityInsightfulMetrics.monotonically_increasing_duration,
    ActivityInsightfulMetrics.duration_increase_of_process_and_or_related_activities,
]


class MetricsFinEffectInterpretation(str, Enum):
    prolonged = "В следующих этапах процесса отмечена высокая длительность. Максимальный потенциальный финансовым эффект от её снижения"
    monotonically_increasing_duration = "Длительность следующих этапов увеличивается со временем, что может привести в дальнейшем к проблемам в процессе.  Максимальный потенциальный финансовый эффект от остановки роста"
    irregular = "Следующие этапы являются нерегулярными (редкими) и не требуются для успешной реализации процесса. Максимальный потенциальный финансовый эффект при отказе от данных этапов"
    has_loops = "На следующих этапах наблюдается зацикленность процесса. Максимальный потенциальный финансовый эффект от устранения зацикленности"
    tight_bottleneck = "В следующих этапах обнаружен Bottle neck, стабильно тормозящий процесс. Максимальный потенциальный финансовый эффект от его устранения"
    widespread_bottleneck = "В следующих этапах обнаружен Bottle neck, тормозящий процесс из-за высокой вариативности времени этапа в разных экземплярах. Максимальный потенциальный финансовый эффект от его устранения"
    numerous_incident_stagnation = "В следующих этапах обнаружен Bottle neck, возникающий из-за многократных инцидентов. Максимальный потенциальный финансовый эффект от его устранения"
    distinctive_incidents_affect_duration = "В следующих этапах наблюдаются разовые инциденты, приводящие к замедлению процесса. Максимальный потенциальный финансовый эффект от их устранения"
    duration_increase_of_process_and_or_related_activities = "Следующие этапы приводят к росту времени процесса и/или прочих этапов.  Максимальный потенциальный финансовый эффект от их устранения"
    system_errors_stagnation = "На данном этапе процесса возникают ошибки системы, приводящие к замедлению процесса. Максимальный потенциальный финансовый эффект от их устранения"
    system_errors_fail = "На данном этапе процесса возникают критические ошибки системы, приводящие к неуспеху процесса. Максимальный потенциальный финансовый эффект от их устранения"
    system_design_errors_fail = "На данном этапе процесса возникают структурные ошибки, приводящие к неуспеху процесса. Максимальный потенциальный финансовый эффект от их устранения"
    reversal_stagnation = "На данном этапе процесса возникают Сторнирование, приводящие к замедлению процесса. Максимальный потенциальный финансовый эффект от их устранения"
    reversal_fail = "На данном этапе процесса возникают Сторнирование, приводящие к неуспеху процесса. Максимальный потенциальный финансовый эффект от их устранения"

    complex_finance = "Суммарный финансовый эффект от АвтоИнсайтов"

    def __str__(self) -> str:
        return self.value


class FinEffectSummaryFormat(str, Enum):
    fin_amount_indent = " "
    money_unit = "рублей"

    fin_activity_sep = ", "
    fin_activity_indent = "\n\t"

    interpretation_sep = "\n\n"
    complex_effect_sep = "*" * 80 + "\n"

    def __str__(self) -> str:
        return self.value


class MessageToUser(str, Enum):
    algorithm_should_run = (
        "Для вывода результата необходимо сначала вызвать метод apply. Метод будет вызван неявно."
    )
    algorithm_succeeded = "Автоинсайты успешно отработали"
    correct_cluster_eps = """Параметр cluster_eps должен быть float из промежутка (0, 1). Был передан: {}"""
    null_duration_median = "Возникла ошибка {err}. Скорее всего, медиана распределения времен этапа {activity} = 0"
    transitions_require_both_timestamps = "Расчет метрик для переходов этапов не может быть проведен\n\t\t\tДолжны быть переданы начало и конец процесса\n\t\t\tЕсли такой информации нет в данных, вы можете запустить алгоритм только для одиночных этапов"

    def __str__(self) -> str:
        return self.value
