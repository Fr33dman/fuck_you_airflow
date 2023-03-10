## Инструкция по установке библиотеки sberPM

##### Рекомендуемая версия python 3.8-3.9
##### Рекомендуемый дистрибутив анаконды - техническое имя: SA100EA9

#### I. Получение доступа к библиотеке.
- ##### Доступ "Альфа" -> Универсальная заявка на доступ к АС -> АС: "BitBucket CI DF Версионный контроль исходного кода Фабрики Данных"
- ##### Доступ "Сигма" -> Универсальная заявка на доступ к АС -> АС: "STASH Версионный контроль исходного кода (BitBucket CI)"

#### II. Скачивание библиотеки (один из способов).

- При помощи Git.
    1. Отрыть консоль (например, Git Bash) и перейти в директорию, куда будет скачан код.
    2. Склонировать репозиторий, выполнив команду:
       1. Сигма git clone https://sbtatlas.sigma.sbrf.ru/stash/scm/rdb/sberpm.git
       2. Альфа git clone https://df-bitbucket.ca.sbrf.ru/scm/sberpm/sberpm.git

- Без использования Git.
    1. Перейти на главную страницу библиотеки в репозитории  https://sbtatlas.sigma.sbrf.ru/stash/projects/RDB/repos/sberpm/browse (Сигма)
    2. Выбрать ветку release, если она не выбрана.
    3. Открыть меню (три точки справа от названия ветки) и нажать Download.
    4. Распаковать скачанный архив в нужную директорию.

#### III. Проверка и настройка конфигурационных файлов (pip.ini).
Если ранее не работали с установкой библиотек на python в домене Sigma или Alpha, то вам необходимо настроить конфигурационный файл для корректной установки библиотеки в противном случае будут вылезать различные ошибки связанные с установкой внешних библиотек.

##### SberOSC
   Можно воспользоваться решением Sber Open Source Compliance (ссылка Sigma https://confluence.sberbank.ru/pages/viewpage.action?spaceKey=IH&title=SberOSC.+Open+Source+Compliance ). По данной ссылке находится подробное описание по настройке конфигурационного файла.
     
#### IV. Установка библиотеки.
1. Отрыть командную строку (например, Anaconda Promt) и перейти в корневую директорию библиотеки:
    ```
    cd sberpm
    ```
2. Установить библиотеку (одним из способов).
    - Установка в обычном режиме. Выполнить команду:
        ```
        pip install .
        ```
    После этого скачанный код библиотеки, то есть директорию sberpm, можно будет удалить.
    - Установка в режиме редактирования.
        Выполнить команду:
        ```
        pip install -e .
        ```
        При такой установке данную директорию удалять нельзя, так как код, находящийся внутри, и будет использоваться.
   
#### V. Обновление библиотеки.  
1. Скачивание обновлённого кода (одним из способов).
    - Если на компьютере хранится код библиотеки, скачанный при помощи Git, его можно обновить с помощью Git:
        1. Отрыть консоль (например, Git Bash) и перейти в корневую директорию библиотеки.
            ```
            cd sberpm
            ```
        2. Обновить проект, выполнив команду: 
            ```
            git pull
            ```
    - Иначе нужно будет скачать новый код любым из способов пункта I.
                  
2. Установка обновлённого кода (одним из способов).
    - Если установка была произведена в обычном режиме, нужно установить обновлённую библиотеку таким же образом (в обычном режиме).
    - Если установка была произведена в режиме редактирования, нужно заменить папку проекта sberpm со старым кодом на папку проекта с обновлённым кодом. 
    Производить установку в данном случае не нужно.  
    *\* Если код был обновлён при помощи команды "git pull", никаких действий по замене кода делать не требуется.*
        
         
            
#### VI. Установка Graphviz.
Данный набор программ необходим для отрисовки графов. 
Без данного пакета можно будет работать с модулями библиотеки, не связанными с графами.
1. Её можно скачать на домен sigma и alpha корпоративного магазина приложений SberUserSoft введя в строке для поиска название библиотеки. После установки библиотека будет расположена в данной директории 'C:/Program Files (x86)/graphviz/bin'. Либо попробовать скачать её напрямую через pip install graphviz, но в данном варианте скачивания могут возникать ошибки.
2. Необходимо прописать путь к исполняемым файлам Graphviz в переменную окружения PATH. 
    Как правило, это папка bin. Для ориентира - в ней должен находиться файл dot.exe.  
    *\*Если при выполнении кода происходит ошибка из-за невозможности найти исполняемые файлы Graphviz, можно добавить путь в переменную PATH прямо в коде  (вместо 'C:/Program Files (x86)/graphviz/bin' нужно указать реальный путь к исполняемым файлам):*
    
    ``` python
    import os
    os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/graphviz/bin'
    ```
3. Если graphviz установить не получается, можно пользоваться библиотекой без него, 
но будет невозможно отрисовывать графы. 
Однако посчитанные графы можно сохранить и загрузить на компьютере с установленным graphviz:

    ``` python
    graph.save('file_name.pkl')  
    from sberpm.visual import load_graph
    graph = load_graph('file_name.pkl')
    ```
#### VII. Настройка NLTK
Библиотека SberPM имеет зависимость от пакета библиотек и программ для символьной и статистической обработки естественного языка (nltk). И для корректной работы библиотеки с модулями (класс LSASummarizer для Суммаризации, класс TextPreprocessing для Препоцессинга текста) необходимо подгрузить компоненты 'punkt' и 'stopwords.

На при наличии доступа во внешнюю сеть можно воспользоваться следующими командами:
   ``` python
   import nltk
   nltk.download('punkt') # need for LSASummarizer
   nltk.download('stopwords') #need for TextPreprocessing
   ```
При отсутсвии доступа во внешнюю сеть можно воспользоваться вспомогательным файлом (support files), добавленным внутрь репозитория. Необходимо перенести файл nltk_data из текущей папки в один из следующих каталогов:
- "C:\Anaconda3\\"
- "C:\\"
- "C:\Users\\"Имя пользователя"\AppData\Roaming\\"

#### VIII. Установка предобученных моделей.
В библиотеке некоторые модули зависят от наличия предобученных моделей (Navec, PM, Bert). Их можно скачать напрямую с bitbucket, либо запросив у разработчиков библиотеки.
Модели необходимо расположить в следующей директории: sberpm\models.

Проверить наличие моделей можно с помощью вспомогательного модуля библиотеки следующим способом:
``` python 
import sberpm.models as check
check.check_contained_models() #- True - модель имеется, False - модель отсутсвует
check.get_models_path() #- получить путь, где должны храниться модели
```
Обратите внимание на то, что библиотека чувствительна к названиям файлов в которых находятся модели.

##### Без предобученных моделей не будут работать некоторые модули связанных с текстовым анализом.
   
#### IX. Установка библиотек требующих Microsoft visual c++ build tools (Актуально для пользователей Windows)
1. Данную проблему можно обойти используя conda install "Название библиотеки". 
2. Чтобы иметь возможность скачивать библиотеки через conda install, необходимо настроить проксирование через SberOSC (https://confluence.sberbank.ru/pages/viewpage.action?spaceKey=IH&title=SberOSC.+Open+Source+Compliance )

#### Замечание
Библиотека гарантированно работает при версии питона 3.8-3.9.