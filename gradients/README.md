## Алгоритм градиентного спуска

1. Задают начальное приближение и точность расчёта $\vec{x}^0, \varepsilon$

2. Рассчитывают $\vec{x}^{[j+1]}=\vec{x}^{[j]}-\lambda^{[j]}\nabla F\left(\vec{x}^{[j]}\right)$, где $\lambda^{[j]}=\mathrm{argmin}_{\lambda} \,F\left(\vec{x}^{[j]}-\lambda\nabla F\left(\vec{x}^{[j]}\right)\right) $

3. Проверяют условие остановки:
   * Если $\left|\vec{x}^{[j+1]}-\vec{x}^{[j]}\right|>\varepsilon$, $\left|F\left(\vec{x}^{[j+1]}\right)-F\left(\vec{x}^{[j]}\right)\right|>\varepsilon$ или $ \left\| \nabla F\left(\vec{x}^{[j+1]}\right) \right\| > \varepsilon$ (выбирают одно из условий), то $j=j+1$ и переход к шагу 2.
   * Иначе $\vec{x}=\vec{x}^{[j+1]}$ и останов.