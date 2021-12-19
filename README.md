# Домашнее задание по NV
[Условие](https://github.com/markovka17/dla/tree/2021/hw4_nv)

## Установка
```
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Обзор репозитория
1. В папке src/ находится почти весь исходный код.
2. В папке data/ находится разбиение LJSpeech-а на train и val. Это разбиение сгененировано командой `python gen_trainval_split.py`
3. Скрипт train.py запускает обучение на LJSpeech. Все необходимые данные он подгружает, если их нет. Параметры, с которыми я вызывал этот скрипт, можно посмотреть в логах конкретного запуска в wandb.
4. Скрипт test.py тестирует модель на 3-ех заранее выбранных предложениях. Он загружает веса модели автоматически, но для этого нужно указать запуск wandb, который сгенерировал веса, и название файла с весами. Его можно запускать, например, командой `python test.py --wandb-run-path _username_/HiFiGan/runs/3eqc5dvn --wandb-file-name states/state125000.pth`.

## Выполненная работа
1. Написана модель [HiFi-GAN](https://arxiv.org/abs/2010.05646), код можно найти в `src/models`.
2. Выполнено обучение модели. Все логи можно найти [здесь](https://wandb.ai/_username_/HiFiGan). Отчет о проделанной работе можно найти [здесь](https://wandb.ai/_username_/HiFiGan/reports/-NV--VmlldzoxMzU3Mzg3).
3. Модель протестирована на 3-ех предложениях из условия. Результаты можно увидеть [здесь](https://wandb.ai/_username_/HiFiGan/runs/320w90xr).
