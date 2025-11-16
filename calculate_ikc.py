```python
import argparse

def calculate_ikc(cloud, iot, remote, auto):
    ikc = 0.4 * cloud + 0.3 * iot + 0.2 * remote + 0.0 * auto
    print(f"ИКЦ = {ikc:.2f} ({ikc*100:.0f} баллов)")
    if ikc < 0.45:
        print("Зелёная зона – низкий риск")
    elif ikc < 0.65:
        print("Оранжевая зона – умеренный риск")
    else:
        print("Красная зона – критический риск")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Расчёт индекса киберцифровизации (ИКЦ)")
    parser.add_argument("--cloud", type=float, required=True, help="Доля данных в облаке (0–1)")
    parser.add_argument("--iot", type=float, required=True, help="IoT на 100 сотрудников (0–1)")
    parser.add_argument("--remote", type=float, required=True, help="Доля удалённых сотрудников (0–1)")
    parser.add_argument("--auto", type=float, required=True, help="Уровень автоматизации (0–1)")
    args = parser.parse_args()
    calculate_ikc(args.cloud, args.iot, args.remote, args.auto)
