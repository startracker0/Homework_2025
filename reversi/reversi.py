import numpy as np
import random
import math
import time
from copy import deepcopy

class Board:
    """黑白棋棋盘类"""
    # 方向数组，用于检查8个方向上的棋子
    DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    
    def __init__(self, size=8):
        """初始化棋盘"""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        # 初始棋盘设置
        mid = size // 2
        self.board[mid-1][mid-1] = 1  # 白棋
        self.board[mid][mid] = 1      # 白棋
        self.board[mid-1][mid] = -1   # 黑棋
        self.board[mid][mid-1] = -1   # 黑棋
        self.current_player = -1      # 黑棋先手
        
    def get_valid_moves(self, player=None):
        """获取当前玩家的所有有效落子位置"""
        if player is None:
            player = self.current_player
            
        valid_moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0 and self.would_flip(i, j, player):
                    valid_moves.append((i, j))
        return valid_moves
    
    def would_flip(self, row, col, player):
        """检查落子在(row, col)是否会翻转对手的棋子"""
        if self.board[row][col] != 0:
            return False
            
        for dr, dc in self.DIRECTIONS:
            r, c = row + dr, col + dc
            if not (0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == -player):
                continue
                
            r, c = r + dr, c + dc
            while 0 <= r < self.size and 0 <= c < self.size:
                if self.board[r][c] == 0:
                    break
                if self.board[r][c] == player:
                    return True
                r, c = r + dr, c + dc
        return False
    
    def make_move(self, row, col, player=None):
        """在指定位置落子并翻转棋子"""
        if player is None:
            player = self.current_player
            
        if not self.would_flip(row, col, player):
            return False
            
        self.board[row][col] = player
        flipped = self.flip_pieces(row, col, player)
        self.current_player = -player
        
        # 如果下一个玩家没有合法移动，轮到对手
        if not self.get_valid_moves():
            self.current_player = -self.current_player
            # 如果对手也没有合法移动，游戏结束
            if not self.get_valid_moves():
                self.current_player = 0  # 游戏结束
                
        return flipped
    
    def flip_pieces(self, row, col, player):
        """翻转被夹住的对手棋子"""
        flipped = []
        for dr, dc in self.DIRECTIONS:
            to_flip = []
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == -player:
                to_flip.append((r, c))
                r, c = r + dr, c + dc
                
            if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                for fr, fc in to_flip:
                    self.board[fr][fc] = player
                    flipped.append((fr, fc))
                    
        return flipped
    
    def get_score(self):
        """计算当前分数"""
        black = np.sum(self.board == -1)
        white = np.sum(self.board == 1)
        return {'black': black, 'white': white}
    
    def is_game_over(self):
        """判断游戏是否结束"""
        if np.sum(self.board == 0) == 0:  # 棋盘已满
            return True
        # 双方都没有合法移动
        if not self.get_valid_moves(-1) and not self.get_valid_moves(1):
            return True
        return False
    
    def print_board(self):
        """打印当前棋盘状态"""
        print("  ", end="")
        for i in range(self.size):
            print(f" {i}", end="")
        print()
        
        for i in range(self.size):
            print(f"{i} ", end="")
            for j in range(self.size):
                if self.board[i][j] == 0:
                    print("· ", end="")
                elif self.board[i][j] == 1:
                    print("○ ", end="")
                else:
                    print("● ", end="")
            print()
        
        score = self.get_score()
        print(f"黑棋(●): {score['black']}  白棋(○): {score['white']}")
        if self.current_player == -1:
            print("轮到黑棋(●)")
        elif self.current_player == 1:
            print("轮到白棋(○)")


class MCTSNode:
    """蒙特卡洛树搜索节点"""
    def __init__(self, board, parent=None, move=None):
        self.board = deepcopy(board)  # 当前棋盘状态
        self.parent = parent          # 父节点
        self.move = move              # 到达此节点的移动
        self.children = []            # 子节点
        self.wins = 0                 # 胜利次数
        self.visits = 0               # 访问次数
        self.untried_moves = self.board.get_valid_moves()  # 未尝试的移动
        self.player = board.current_player  # 当前玩家
    
    def select_child(self):
        """使用UCB1公式选择最有希望的子节点"""
        # UCB1 = wins/visits + C * sqrt(ln(parent_visits)/visits)
        C = 1.41  # 探索参数
        
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            # 避免除以零
            if child.visits == 0:
                score = float('inf')
            else:
                score = child.wins / child.visits + C * math.sqrt(2 * math.log(self.visits) / child.visits)
                
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    
    def expand(self):
        """扩展一个新的子节点"""
        if not self.untried_moves:  # 没有未尝试的移动
            return None
            
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)
        
        new_board = deepcopy(self.board)
        new_board.make_move(move[0], move[1])
        
        child_node = MCTSNode(new_board, parent=self, move=move)
        self.children.append(child_node)
        return child_node
    
    def simulate(self):
        """从当前节点随机模拟到游戏结束"""
        simulation_board = deepcopy(self.board)
        current_player = simulation_board.current_player
        
        while not simulation_board.is_game_over():
            valid_moves = simulation_board.get_valid_moves()
            if not valid_moves:
                simulation_board.current_player = -simulation_board.current_player
                continue
                
            # 随机选择移动
            random_move = random.choice(valid_moves)
            simulation_board.make_move(random_move[0], random_move[1])
            
        # 游戏结束，计算结果
        score = simulation_board.get_score()
        if score['black'] > score['white']:
            return -1  # 黑棋赢
        elif score['white'] > score['black']:
            return 1   # 白棋赢
        else:
            return 0   # 平局
    
    def backpropagate(self, result):
        """反向传播模拟结果"""
        self.visits += 1
        if self.player == -result:
            self.wins += 1
            
        if self.parent:
            self.parent.backpropagate(result)
    
    def is_fully_expanded(self):
        """检查是否所有可能的移动都已扩展"""
        return len(self.untried_moves) == 0


class MCTS:
    """蒙特卡洛树搜索算法"""
    def __init__(self, board, time_limit=1.0):
        self.root = MCTSNode(board)
        self.time_limit = time_limit
    
    def search(self):
        """执行蒙特卡洛树搜索"""
        end_time = time.time() + self.time_limit
        
        while time.time() < end_time:
            # 1. 选择
            node = self.select()
            # 2. 扩展
            if not node.is_fully_expanded() and not node.board.is_game_over():
                node = node.expand()
            # 3. 模拟
            result = node.simulate()
            # 4. 回溯
            node.backpropagate(result)
        
        # 返回访问次数最多的子节点对应的移动
        best_child = self.get_best_move()
        return best_child.move
    
    def select(self):
        """从根节点选择到叶节点"""
        node = self.root
        
        while node.is_fully_expanded() and node.children:
            node = node.select_child()
            
        return node
    
    def get_best_move(self):
        """获取最佳移动(访问次数最多的子节点)"""
        best_visits = -1
        best_child = None
        
        for child in self.root.children:
            if child.visits > best_visits:
                best_visits = child.visits
                best_child = child
                
        return best_child


def play_game():
    """开始人机对战游戏"""
    board = Board()
    board.print_board()
    
    # 玩家选择棋子颜色
    player_color = input("请选择棋子颜色 (黑棋: b, 白棋: w): ").strip().lower()
    if player_color == 'w':
        player = 1   # 白棋
        ai = -1      # 黑棋
    else:
        player = -1  # 黑棋
        ai = 1       # 白棋
    
    # 游戏主循环
    while not board.is_game_over():
        if board.current_player == player:
            # 玩家回合
            valid_moves = board.get_valid_moves()
            if not valid_moves:
                print("你没有合法移动，跳过回合")
                board.current_player = -board.current_player
                continue
                
            print("合法移动:", valid_moves)
            try:
                row, col = map(int, input("请输入你的移动 (行 列): ").strip().split())
                if (row, col) not in valid_moves:
                    print("非法移动，请重试!")
                    continue
            except ValueError:
                print("输入格式错误，请输入两个数字!")
                continue
                
            board.make_move(row, col)
        else:
            # AI回合
            print("AI思考中...")
            mcts = MCTS(board, time_limit=2.0)
            move = mcts.search()
            print(f"AI移动: {move}")
            board.make_move(move[0], move[1])
            
        board.print_board()
    
    # 游戏结束，显示结果
    score = board.get_score()
    if score['black'] > score['white']:
        winner = "黑棋(●)"
    elif score['white'] > score['black']:
        winner = "白棋(○)"
    else:
        winner = "平局"
    print(f"游戏结束! 胜者: {winner}")


if __name__ == "__main__":
    play_game()