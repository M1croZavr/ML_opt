## Алгоритм градиентного спуска

1. Задают начальное приближение и точность расчёта <img src="https://render.githubusercontent.com/render/math?math=\vec{x}^0, \varepsilon">

2. Рассчитывают <img src="https://render.githubusercontent.com/render/math?math=\vec{x}^{[j+1]}=\vec{x}^{[j]}-\lambda^{[j]}\nabla F\left(\vec{x}^{[j]}\right)">, где <img src="https://render.githubusercontent.com/render/math?math=\lambda^{[j]}=\mathrm{argmin}_{\lambda} \,F\left(\vec{x}^{[j]}-\lambda\nabla F\left(\vec{x}^{[j]}\right)\right) ">

3. Проверяют условие остановки:
   * Если <img src="https://render.githubusercontent.com/render/math?math=\left|\vec{x}^{[j+1]}-\vec{x}^{[j]}\right|>\varepsilon">, <img src="https://render.githubusercontent.com/render/math?math=\left|F\left(\vec{x}^{[j+1]}\right)-F\left(\vec{x}^{[j]}\right)\right|>\varepsilon"> или <img src="https://render.githubusercontent.com/render/math?math=\left\| \nabla F\left(\vec{x}^{[j+1]}\right) \right\| > \varepsilon"> (выбирают одно из условий), то <img src="https://render.githubusercontent.com/render/math?math=j=j%2B1"> и переход к шагу 2.
   * Иначе <img src="https://render.githubusercontent.com/render/math?math=\vec{x}=\vec{x}^{[j%2B1]}"> и останов.

