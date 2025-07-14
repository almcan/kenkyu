from mesa import Model
from mesa.space import SingleGrid
from mesa.time import RandomActivation

class BlueToothWormModel(Model):
    """
    BlueToothワームの拡散モデル
    """
    def __init__(self, width=100, height=100):
        super().__init__()
        self.grid = SingleGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)

# モデルの実行
if __name__ == "__main__":
    model = BlueToothWormModel(width=100, height=100)
    print("モデルの作成に成功しました。")
    print(f"グリッドの幅: {model.grid.width}, グリッドの高さ: {model.grid.height}")