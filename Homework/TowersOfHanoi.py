from copy import copy


class ToHState:
    """
    Represents the state of the Towers of Hanoi game.
    """

    towers_names = ['A', 'B', 'C']
    discs = [1, 2, 3]

    def __init__(self, a, b, c, parent=None, action=""):
        self.towers = [copy(a), copy(b), copy(c)] # state
        self.parent = parent # previous state
        self.action = action # string description of step that lead to this state

    def test(self) -> bool:
        """
        Checks if the current state adheres to the rules of the game.

        Returns:
            bool: True if valid, False otherwise.
        """

        for tower in self.towers:
            last = max(self.discs) + 1
            for disc in tower:
                if disc > last:
                    return False
                last = disc
        return True

    def next_states(self) -> list:
        """
        Generates all valid states reachable from this state.

        Returns:
            list: List of valid next states.
        """

        states = []
        for idx, tower in enumerate(self.towers):
            if tower:
                new_tower = copy(tower)
                move = new_tower.pop(-1)
                available_towers = [t for t in range(len(self.towers)) if t != idx]
                for available_tower in available_towers:
                    new_state_towers = copy(self.towers)
                    new_state_towers[idx] = new_tower
                    new_state_towers[available_tower] = self.towers[available_tower] + [move]
                    new_state = ToHState(*new_state_towers, parent=self, action=f"Move disk {move} from {self.towers_names[idx]} to {self.towers_names[available_tower]}")
                    if new_state.test():
                        states.append(new_state)
        return states

    def ancestors(self) -> list:
        """
        Returns a list of all ancestors of this state.

        Returns:
            list: List of ancestor states.
        """

        if self.parent is None:
            return []
        else:
            return [self.parent] + self.parent.ancestors()

    def __repr__(self):
        return f'Action="{self.action}", ToHState(a={self.towers[0]}, b={self.towers[1]}, c={self.towers[2]})'

    def __eq__(self, b):
        return self.towers[0] == b.towers[0] and self.towers[1] == b.towers[1] and self.towers[2] == b.towers[2]


def breadth_first_search(start: ToHState, final: ToHState) -> ToHState:
    """
    Performs breadth-first search to find a solution from the start state to the final state.

    Args:
        start: The initial state.
        final: The target state.

    Returns:
        ToHState: The solution state (if found), otherwise None.
    """

    seen = [start] # List of already visited states
    open = start.next_states() # Buffer of discovered states
    solution = None  # Final state that matches the goal

    while open:
        current = open.pop(0)
        if current == final:
            return current
        if current not in seen:
            seen.append(current)
            open += current.next_states()

    return solution


def depth_first_search(start: ToHState, final: ToHState) -> ToHState:
    """
    Performs depth-first search to find a solution from the start state to the final state.

    Args:
        start: The initial state.
        final: The target state.

    Returns:
        ToHState: The solution state (if found), otherwise None.
    """

    seen = [start]  # List of already visited states
    open = start.next_states()  # Buffer of discovered states
    solution = None  # Final state that matches the goal

    while open:
        current = open.pop(0)
        if current == final:
            return current
        if current not in seen:
            seen.append(current)
            open = current.next_states() + open

    return solution



# TEST
start = ToHState([3,2,1], [], [], action="START")
final = ToHState([], [], [3, 2, 1]) # goal

solution = breadth_first_search(start, final)

if solution:
    print('breadth-first search solution found')
    way = [solution] + solution.ancestors()
    for step in way[-1::-1]:
        print(step)

else:
    print('breadth-first search solution not found')

solution = depth_first_search(start, final)

if solution:
    print('depth-first search solution found')
    way = [solution] + solution.ancestors()
    for step in way[-1::-1]:
        print(step)

else:
    print('depth-first search solution not found')
