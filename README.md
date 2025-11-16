# LSTM-модель прогнозирования инцидентов ИБ при цифровизации

Репозиторий к статье  
**«Анализ эволюции угроз информационной безопасности в условиях цифровизации: авторская методология и модель прогнозирования»**  
Алексеев В.П., Сиротина Д.С., Кулешова А.В., 2025

## Содержание
- `model.py` — полная архитектура и обучение модели
- `calculate_ikc.py` — расчёт индекса киберцифровизации (ИКЦ)
- `generate_synthetic_data.py` — генерация синтетических данных для теста
- `requirements.txt` — зависимости
- `nastiapros_forecast_best.h5` — готовые веса модели (загружаются автоматически)

## Быстрый старт

```bash
git clone https://github.com/nastiapros/lstm-ib-forecast.git
cd lstm-ib-forecast
pip install -r requirements.txt
python model.py          # обучение и прогноз
python calculate_ikc.py --cloud 0.8 --iot 0.5 --remote 0.6 --auto 0.5
