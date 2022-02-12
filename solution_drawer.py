import matplotlib.pyplot as plt

class SolutionDrawer:
    @staticmethod
    def get_cmap(n, name='hsv'):
        return plt.cm.get_cmap(name, n)

    @staticmethod
    def draw(solution, customers):
        plt.clf()
        SolutionDrawer.draw_points(customers)
        SolutionDrawer.draw_routes(solution)
        plt.savefig('solution')

    @staticmethod
    def draw_points(customers):
        x = []
        y = []
        for cust in customers:
            x.append(cust.x)
            y.append(cust.y)
        plt.scatter(x, y, c="blue")

    @staticmethod
    def draw_routes(solution):
        cmap = SolutionDrawer.get_cmap(len(solution.routes))
        if solution is None:
            return
        
        for i, route in enumerate(solution.routes):
            for j in range(len(route) - 1):
                cust_1 = route[j]
                cust_2 = route[j + 1]
                plt.plot([cust_1.x, cust_2.x], [cust_1.y, cust_2.y], c=cmap(i))
