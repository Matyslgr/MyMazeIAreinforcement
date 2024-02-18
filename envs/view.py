import pygame
import numpy as np

class View:

    def __init__(self, maze_name="Maze2D", maze_size=(30, 30), screen_size=(1800, 1400),
                 maze_screen=(1400, 1400), has_loops=False, num_portals=0, entrance=None,
                 goal=None, maze=None, optimise=False, time=0):

        pygame.display.set_caption(maze_name)
        self.clock = pygame.time.Clock()
        self.__game_over = False
        self.maze_screen = maze_screen

        self.maze_size = maze_size
        self.__maze = maze
        self.__robot = np.zeros(2, dtype=int)

        # to show the right and bottom border
        self.screen = pygame.display.set_mode(screen_size)
        self.screen_size = tuple(map(sum, zip(screen_size, (-1, -1))))
        
        self.__entrance = entrance if entrance is not None else np.zeros(2, dtype=int)
        self.__goal = goal if goal is not None else np.array(self.maze_size) - np.array((1, 1))

        # The path
        self.path_matrix = np.zeros_like(self.maze.maze_cells)

        # Create a background
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill((255, 255, 255))
        # Create a layer for the maze
        self.maze_layer = pygame.Surface(self.maze_screen).convert_alpha()
        self.maze_layer.fill((0, 0, 0, 0,))
        # show the maze
        self.__draw_maze()
        # show the portals
        self.__draw_portals()
        # show the robot
        self.__draw_robot()
        # show the entrance
        self.__draw_entrance()
        # show the goal
        self.__draw_goal()
        
        self.optimise = optimise
        self.mid_right = (screen_size[0] - maze_screen[0]) / 2 + maze_screen[0]
        width_button = self.mid_right / 4
        height_button = maze_screen[1] / 10
        self.padding_top = height_button / 2
        self.train_button_rect = pygame.Rect(self.mid_right - width_button / 2, self.padding_top, width_button, height_button)
        self.simulate_button_rect = pygame.Rect(self.mid_right - width_button / 2, self.padding_top * 2 + height_button, width_button, height_button)
        self.train_button_color = (150, 255, 150)  # Exemple de couleur plus douce pour le bouton "Train"
        self.simulate_button_color = (255, 150, 150)  # Exemple de couleur plus douce pour le bouton "Simulate"
        self.train_button_text = "Train"
        self.simulate_button_text = "Simulate"
        self.font = pygame.font.Font("Font/GROBOLD.ttf", 36)
        self.train_text = self.font.render(self.train_button_text, True, (0, 0, 0))
        self.simulate_text = self.font.render(self.simulate_button_text, True, (0, 0, 0))

        self.text_train_center = (self.train_button_rect.center[0] - self.train_text.get_width() / 2, self.train_button_rect.center[1] - self.train_text.get_height() / 2)
        self.text_simulate_center = (self.simulate_button_rect.center[0] - self.simulate_text.get_width() / 2, self.simulate_button_rect.center[1] - self.simulate_text.get_height() / 2)
        # show buttons
        self.total_reward = 0
        self.__draw_buttons()
        
        # show texts
        if self.optimise:
            self.time = time
            self.text_time = self.font.render("Time: %.3f" % self.time, True, (0, 0, 0))
            self.text_time_center = (self.mid_right - self.text_time.get_width() / 2, self.maze_screen[1] / 2 - self.text_time.get_height() / 2 + self.padding_top * 2)
        self.draw_texts()
        

    def draw_shortest_path(self):
        if self.maze.shortest_path is None:
            return
        path_color = (255, 224, 134)
        for cell in self.maze.shortest_path:
            self.__colour_cell(cell, colour=path_color, transparency=200)

    def reset_path_matrix(self):
        self.path_matrix = np.zeros_like(self.maze.maze_cells)

    def update(self, mode="human", position=None, total_reward=0):
        try:
            self.set_robot(position) if position is not None else None
            self.path_matrix[self.robot[0], self.robot[1]] = 1
            self.set_reward(total_reward)
            self.__view_update(mode)
            return self.__controller_update()
        except Exception as e:
            self.__game_over = True
            self.quit_game()
            raise e

    def quit_game(self):
        try:
            self.__game_over = True
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    def set_color_text(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.train_button_rect.collidepoint(mouse_pos):
            self.train_button_color = (0, 200, 0)
        else:
            self.train_button_color = (150, 255, 150)

        if self.simulate_button_rect.collidepoint(mouse_pos):
            self.simulate_button_color = (200, 0, 0)
        else:
            self.simulate_button_color = (255, 150, 150)

    def set_reward(self, reward):
        self.total_reward = reward

    def __controller_update(self):
        if not self.__game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.__game_over = True
                    self.quit_game()
                    return "exit"
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.optimise and self.train_button_rect.collidepoint(event.pos):
                        return "train"
                    elif self.simulate_button_rect.collidepoint(event.pos):
                        return "simulate"
            self.set_color_text()

    def __view_update(self, mode="human"):
        if not self.__game_over:
            # update the robot's position
            self.__draw_path()
            self.draw_shortest_path()
            self.__draw_entrance()
            self.__draw_robot()
            self.__draw_goal()
            self.__draw_portals()
            self.__draw_maze()

            # update the screen
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.maze_layer,(0, 0))
            self.draw_texts()
            self.__draw_buttons()
            if mode == "human":
                pygame.display.flip()

    def __draw_maze(self):
        
        line_colour = (0, 0, 0, 255)

        # drawing the horizontal lines
        for y in range(self.maze.MAZE_H + 1):
            pygame.draw.line(self.maze_layer, line_colour, (0, y * self.CELL_H),
                             (self.MAZE_SCREEN_W, y * self.CELL_H))

        # drawing the vertical lines
        for x in range(self.maze.MAZE_W + 1):
            pygame.draw.line(self.maze_layer, line_colour, (x * self.CELL_W, 0),
                             (x * self.CELL_W, self.MAZE_SCREEN_H))

        # drawing the vertical lines at the end
        pygame.draw.line(self.maze_layer, line_colour, (self.MAZE_SCREEN_W - 2, 0),
                         (self.MAZE_SCREEN_W - 1, self.MAZE_SCREEN_H))

        # drawing the horizontal lines at the end
        pygame.draw.line(self.maze_layer, line_colour, (0, self.MAZE_SCREEN_H - 2),
                         (self.MAZE_SCREEN_W, self.MAZE_SCREEN_H - 2))
        # breaking the walls
        for x in range(len(self.maze.maze_cells)):
            for y in range (len(self.maze.maze_cells[x])):
                # check the which walls are open in each cell
                walls_status = self.maze.get_walls_status(self.maze.maze_cells[x, y])
                dirs = ""
                for dir, open in walls_status.items():
                    if open:
                        dirs += dir
                self.__cover_walls(x, y, dirs)

    def __cover_walls(self, x, y, dirs, colour=(0, 0, 255, 50)):
        
        dx = x * self.CELL_W
        dy = y * self.CELL_H

        if not isinstance(dirs, str):
            raise TypeError("dirs must be a str.")

        for dir in dirs:
            if dir == "S":
                line_head = (dx + 1, dy + self.CELL_H)
                line_tail = (dx + self.CELL_W - 1, dy + self.CELL_H)
            elif dir == "N":
                line_head = (dx + 1, dy)
                line_tail = (dx + self.CELL_W - 1, dy)
            elif dir == "W":
                line_head = (dx, dy + 1)
                line_tail = (dx, dy + self.CELL_H - 1)
            elif dir == "E":
                line_head = (dx + self.CELL_W, dy + 1)
                line_tail = (dx + self.CELL_W, dy + self.CELL_H - 1)
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")

            pygame.draw.line(self.maze_layer, colour, line_head, line_tail)

    def __draw_robot(self, colour=(0, 0, 0), transparency=255):
        
        x = int(self.__robot[0] * self.CELL_W + self.CELL_W * 0.5 + 0.5)
        y = int(self.__robot[1] * self.CELL_H + self.CELL_H * 0.5 + 0.5)
        r = int(min(self.CELL_W, self.CELL_H)/5 + 0.5)
        pygame.draw.circle(self.maze_layer, colour + (transparency,), (x, y), r)

    def __draw_path(self):

        for x in range(self.maze.MAZE_W):
            for y in range(self.maze.MAZE_H):
                if self.path_matrix[x, y] == 1:
                    self.__colour_cell((x, y), colour=(225, 225, 225), transparency=180)
                else:
                    self.__colour_cell((x, y), colour=(255, 255, 255), transparency=255)

    def __draw_entrance(self, colour=(0, 0, 150), transparency=235):

        self.__colour_cell(self.entrance, colour=colour, transparency=transparency)

    def __draw_goal(self, colour=(150, 0, 0), transparency=235):

        self.__colour_cell(self.goal, colour=colour, transparency=transparency)

    def __draw_portals(self, transparency=160):
        
        colour_range = np.linspace(0, 255, len(self.maze.portals), dtype=int)
        colour_i = 0
        for portal in self.maze.portals:
            colour = ((100 - colour_range[colour_i])% 255, colour_range[colour_i], 0)
            colour_i += 1
            for location in portal.locations:
                self.__colour_cell(location, colour=colour, transparency=transparency)

    def __colour_cell(self, cell, colour, transparency):

        if not (isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 2):
            raise TypeError("cell must a be a tuple, list, or numpy array of size 2")

        x = int(cell[0] * self.CELL_W + 0.5 + 1)
        y = int(cell[1] * self.CELL_H + 0.5 + 1)
        w = int(self.CELL_W + 0.5 - 1)
        h = int(self.CELL_H + 0.5 - 1)
        pygame.draw.rect(self.maze_layer, colour + (transparency,), (x, y, w, h))

    def __draw_buttons(self):
        
        pygame.draw.rect(self.screen, self.simulate_button_color, self.simulate_button_rect)
        self.screen.blit(self.simulate_text, self.text_simulate_center)
        if not self.optimise:
            pygame.draw.rect(self.screen, self.train_button_color, self.train_button_rect)
            self.screen.blit(self.train_text, self.text_train_center)

    def draw_texts(self):
        text = self.font.render("Reward: %.3f" % self.total_reward, True, (0, 0, 0))
        text_center = (self.mid_right - text.get_width() / 2, self.maze_screen[1] / 2 - text.get_height() / 2)
        
        if self.optimise:
            self.screen.blit(self.text_time, self.text_time_center)
        self.screen.blit(text, text_center)

    def render(self, mode="human", close=False, position=None, total_reward=0):
        if close:
            self.quit_game()
        return self.update(mode, position=position, total_reward=total_reward)

    @property
    def robot(self):
        return self.__robot

    def set_robot(self, position):
        self.__robot = position

    @property
    def maze(self):
        return self.__maze

    @property
    def entrance(self):
        return self.__entrance

    @property
    def goal(self):
        return self.__goal

    @property
    def game_over(self):
        return self.__game_over

    @property
    def SCREEN_SIZE(self):
        return tuple(self.screen_size)

    @property
    def MAZE_SCREEN(self):
        return tuple(self.maze_screen)

    @property
    def MAZE_SCREEN_W(self):
        return int(self.MAZE_SCREEN[0])
    
    @property
    def MAZE_SCREEN_H(self):
        return int(self.MAZE_SCREEN[1])

    @property
    def SCREEN_W(self):
        return int(self.SCREEN_SIZE[0])

    @property
    def SCREEN_H(self):
        return int(self.SCREEN_SIZE[1])

    @property
    def CELL_W(self):
        return float(self.MAZE_SCREEN_W) / float(self.maze.MAZE_W)

    @property
    def CELL_H(self):
        return float(self.MAZE_SCREEN_H) / float(self.maze.MAZE_H)
