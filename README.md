# RUN

requires python3.8

copy airflow.cfg to airflow root package

```Shell
pip install -r req.txt
airflow standalone
```


## Trigger dags

### Via API

```
curl --location --request POST '127.0.0.1:8080/api/v1/dags/test_pm_pipeline/dagRuns' \
--header 'Authorization: Basic YWRtaW46VXhVZXpQUGVONEVVOHhnWQ==' \
--header 'Content-Type: application/json' \
--header 'Cookie: session=89b45873-eee6-4542-96f5-11818dc09534.absoyDNonEGnLRVuy58mVG1-wvQ' \
--data-raw '{
    "conf": {
        "python_models": {
            "encode": "UTF-8",
            "separator": ";",
            "datePattern": null,
            "sourceFile": "td0002383_1465",
            "databaseName": "_03ResearchData",
            "notationData": [
                {
                    "name": "_isCaseOutlierMode_1",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Экземпляр. Является выбросом. Настройки выбросов 1",
                    "fieldType": "CASE",
                    "physical": true
                },
                {
                    "name": "_isActivityOutlierMode_1",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Операция. Является выбросом. Настройки выбросов 1",
                    "fieldType": "ACTIVITY",
                    "physical": true
                },
                {
                    "name": "_activityDurationForwardClear",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "DURATION_CLEAR",
                    "aliasName": "Операция. Чистая длительность с учетом перехода",
                    "fieldType": "ACTIVITY",
                    "physical": true
                },
                {
                    "name": "_caseDurationClear",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "DURATION_CLEAR",
                    "aliasName": "Экземпляр. Чистая длительность",
                    "fieldType": "CASE",
                    "physical": true
                },
                {
                    "name": "_caseActivityDurationClear",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "DURATION_CLEAR",
                    "aliasName": "Экземпляр. Чистая длительность операций с учетом переходов",
                    "fieldType": "CASE",
                    "physical": true
                },
                {
                    "name": "\"Чистая_длительность\"",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Чистая длительность",
                    "fieldType": "DATASET",
                    "physical": true
                },
                {
                    "name": "\"Грязная_длительность\"",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Грязная длительность",
                    "fieldType": "DATASET",
                    "physical": true
                },
                {
                    "name": "\"Пользователь\"",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "STRING",
                    "aliasName": "Пользователь",
                    "fieldType": "DATASET",
                    "physical": true
                },
                {
                    "name": "\"Продукт\"",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "STRING",
                    "aliasName": "Продукт",
                    "fieldType": "DATASET",
                    "physical": true
                },
                {
                    "name": "\"Дата_окончания_этапа\"",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "DATE",
                    "aliasName": "Дата окончания этапа",
                    "fieldType": "DATASET",
                    "physical": true
                },
                {
                    "name": "__transitionPercent",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "PERCENT",
                    "aliasName": "Процент переходов",
                    "fieldType": "METRIC",
                    "physical": false
                },
                {
                    "name": "__caseDurationFromStart",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "DURATION",
                    "aliasName": "Длительность от начала экземпляра до окончания операции",
                    "fieldType": "METRIC",
                    "physical": false
                },
                {
                    "name": "_activityDurationForward",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "DURATION",
                    "aliasName": "Операция. Длительность с учетом перехода",
                    "fieldType": "ACTIVITY",
                    "physical": true
                },
                {
                    "name": "_activityEndTimeForward",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "DATE",
                    "aliasName": "Операция. Дата окончания",
                    "fieldType": "ACTIVITY",
                    "physical": true
                },
                {
                    "name": "_childActivity",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "STRING",
                    "aliasName": "Операция. Дочерняя операция",
                    "fieldType": "ACTIVITY",
                    "physical": true
                },
                {
                    "name": "_childActivityId",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Операция. Техн.Идентификатор дочерней операции",
                    "fieldType": "ACTIVITY",
                    "physical": true
                },
                {
                    "name": "_activityStartTime",
                    "keyType": "DATE",
                    "dataType": "DATE",
                    "aliasName": "Операция. Дата начала (Дата_начала_этапа)",
                    "fieldType": "ACTIVITY",
                    "physical": true
                },
                {
                    "name": "__transitionCount",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Количество переходов",
                    "fieldType": "METRIC",
                    "physical": false
                },
                {
                    "name": "__reworkActivityCnt",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Количество повторных операций",
                    "fieldType": "METRIC",
                    "physical": false
                },
                {
                    "name": "__activityCyclePercent",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "PERCENT",
                    "aliasName": "Процент зацикленности операций",
                    "fieldType": "METRIC",
                    "physical": false
                },
                {
                    "name": "_isReworkActivity",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Операция. Является повторной",
                    "fieldType": "ACTIVITY",
                    "physical": true
                },
                {
                    "name": "_activityReworkNum",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Операция. Номер в цикле",
                    "fieldType": "ACTIVITY",
                    "physical": true
                },
                {
                    "name": "_isFirstActivity",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Операция. Является первой",
                    "fieldType": "ACTIVITY",
                    "physical": true
                },
                {
                    "name": "_isLastActivity",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Операция. Является последней",
                    "fieldType": "ACTIVITY",
                    "physical": true
                },
                {
                    "name": "(case when _transitionCnt> _transitionCntUnique then '\''Да'\'' else '\''Нет'\'' end) ",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "STRING",
                    "aliasName": "Путь. Наличие зацикленности по переходам",
                    "fieldType": "CASE",
                    "physical": false
                },
                {
                    "name": "(case when  _activityCnt > _activityCntUnique then '\''Да'\'' else '\''Нет'\'' end )",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "STRING",
                    "aliasName": "Путь. Наличие зацикленности по операциям",
                    "fieldType": "CASE",
                    "physical": false
                },
                {
                    "name": "__caseCount",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Количество экземпляров",
                    "fieldType": "METRIC",
                    "physical": false
                },
                {
                    "name": "__transitionCyclePercent",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "PERCENT",
                    "aliasName": "Процент зацикленности переходов",
                    "fieldType": "METRIC",
                    "physical": false
                },
                {
                    "name": "__activityCount",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Количество операций",
                    "fieldType": "METRIC",
                    "physical": false
                },
                {
                    "name": "_transitionCntUnique",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Путь. Количество уникальных переходов",
                    "fieldType": "CASE",
                    "physical": true
                },
                {
                    "name": "_activityCntUnique",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Путь. Количество уникальных операций",
                    "fieldType": "CASE",
                    "physical": true
                },
                {
                    "name": "_transitionCnt",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Путь. Количество переходов",
                    "fieldType": "CASE",
                    "physical": true
                },
                {
                    "name": "_activityCnt",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Путь. Количество операций",
                    "fieldType": "CASE",
                    "physical": true
                },
                {
                    "name": "_caseActivityDuration",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "DURATION",
                    "aliasName": "Экземпляр. Длительность операций с учетом переходов",
                    "fieldType": "CASE",
                    "physical": true
                },
                {
                    "name": "_caseDuration",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "DURATION",
                    "aliasName": "Экземпляр. Длительность",
                    "fieldType": "CASE",
                    "physical": true
                },
                {
                    "name": "_caseEndTime",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "DATE",
                    "aliasName": "Экземпляр. Дата окончания",
                    "fieldType": "CASE",
                    "physical": true
                },
                {
                    "name": "_caseStartTime",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "DATE",
                    "aliasName": "Экземпляр. Дата начала",
                    "fieldType": "CASE",
                    "physical": true
                },
                {
                    "name": "_variantId",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "STRING",
                    "aliasName": "Экземпляр. Идентификатор пути",
                    "fieldType": "CASE",
                    "physical": true
                },
                {
                    "name": "_activity",
                    "keyType": "STATUS_NAME",
                    "dataType": "STRING",
                    "aliasName": "Операция (Наименование_этапа_процесса)",
                    "fieldType": "ACTIVITY",
                    "physical": true
                },
                {
                    "name": "_case",
                    "keyType": "ID",
                    "dataType": "STRING",
                    "aliasName": "Экземпляр (Идентификатор_экземпляра_процесса)",
                    "fieldType": "CASE",
                    "physical": true
                },
                {
                    "name": "_activityId",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Операция. Техн.Идентификатор операции",
                    "fieldType": "ACTIVITY",
                    "physical": true
                },
                {
                    "name": "_activityNum",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Операция. Номер в экземпляре",
                    "fieldType": "ACTIVITY",
                    "physical": true
                },
                {
                    "name": "_caseId",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "STRING",
                    "aliasName": "Экземпляр. Техн.Идентификатор экземпляра",
                    "fieldType": "CASE",
                    "physical": true
                },
                {
                    "name": "__variantCount",
                    "keyType": "WITHOUT_KEY",
                    "dataType": "NUMERIC",
                    "aliasName": "Количество путей",
                    "fieldType": "METRIC",
                    "physical": false
                }
            ],
            "categoricalFilters": null,
            "intervalFilter": [],
            "idFilter": null,
            "pathFilters": null,
            "cycleFilterType": "DEFAULT",
            "storageFormat": "CLICKHOUSE",
            "widgetId": 0,
            "workspaceId": 2566,
            "mlKeys": {
                "modelName": "FACTOR_ANALYSIS",
                "modelLabel": "ydyd",
                "params": {
                    "target_column": "\"Чистая_длительность\"",
                    "type_of_target": "number",
                    "categorical_cols": [
                        "_childActivity"
                    ],
                    "numeric_cols": [
                        "_activityId"
                    ],
                    "date_cols": [
                        "_caseStartTime",
                        "_caseEndTime"
                    ],
                    "extended_search": false,
                    "count_others": false
                }
            }
        },
        "outliers": {
            "database": "_03ResearchData",
            "table_name": "td0002383_1465",
            "metric_type": "_activityDurationForward",
            "sections": [
                "_activity"
            ],
            "outlier_detector_params": {
                "outlier_detector_type": "MANUAL_INPUT",
                "params": {
                    "lower_bound": 0.1,
                    "upper_bound": 0.9
                }
            },
            "min_row_count": 30,
            "exclude_process": true,
            "use_case_branch": false,
            "outlier_num": 1
        }
    }
}'
```
