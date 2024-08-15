import os
import numpy as np
import random
import pygame as p
import chess

class ChessEnv:
    # Maps each chess piece to a unique integer value for internal representation
    piece_map = {
        '--': 0, 'bp': 1, 'bR': 2, 'bN': 3, 'bB': 4, 'bK': 5, 'bQ': 6,
        'wp': 7, 'wR': 8, 'wN': 9, 'wB': 10, 'wK': 11, 'wQ': 12
    }
    
    # Defines the value of each piece for calculating rewards when pieces are captured
    piece_values = {
        'p': 0.1, 'n': 0.3, 'b': 0.3, 'r': 0.5, 'q': 0.9, 'k': 0  # King has no value as it's not captured
    }

    def __init__(self):
        # Initializes the environment, Pygame, and loads images for pieces
        self.reset()
        self.initialize_pygame()
        self.load_images()

    def initialize_pygame(self):
        # Initializes Pygame settings for displaying the chessboard
        p.init()
        self.screen = p.display.set_mode((512, 512))
        p.display.set_caption('Chess DQN Training')
        self.clock = p.time.Clock()
        self.SQUARE_SIZE = 64
        self.colors = [p.Color("white"), p.Color("gray")]
        self.font = p.font.SysFont("Arial", 24)

    def load_images(self):
        # Loads images for each chess piece and scales them to fit on the board
        self.piece_images = {}
        pieces = ['bp', 'bR', 'bN', 'bB', 'bK', 'bQ', 'wp', 'wR', 'wN', 'wB', 'wK', 'wQ']
        for piece in pieces:
            self.piece_images[piece] = p.transform.scale(
                p.image.load(os.path.join("images", f"{piece}.png")), (self.SQUARE_SIZE, self.SQUARE_SIZE)
            )

    def reset(self):
        # Resets the chessboard to its initial state and sets the game to start with white's move
        self.board = chess.Board()
        self.done = False
        self.current_player = 'w'  # White starts
        self.current_reward = 0  # Initialize current reward
        return self.get_state()

    def get_state(self):
        # Returns the current state of the board in FEN (Forsyth-Edwards Notation) format
        return self.board.fen()

    def step(self, action):
        # Applies the selected action (move) to the board and returns the new state, reward, and whether the game is over
        move = self.action_to_move(action)
        self.board.push(move)
        reward = self.calculate_reward(move)
        self.current_reward = reward  # Update current reward
        self.done = self.is_game_over()
        self.current_player = 'b' if self.current_player == 'w' else 'w'
        return self.get_state(), reward, self.done

    def action_to_move(self, action):
        # Converts the action index to an actual chess move from the list of legal moves
        legal_moves = list(self.board.legal_moves)
        return legal_moves[action]

    def calculate_reward(self, move):
        # Calculates the reward based on the result of the move, such as checkmate, capture, or regular move
        reward = 0
        if self.board.is_checkmate():
            reward = 60 if self.board.turn == chess.BLACK else -60  # Assigns high reward for checkmate
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            reward = 0  # No reward for stalemate or draw
        elif self.board.is_capture(move):
            captured_piece = self.board.piece_at(move.to_square)
            if captured_piece:
                piece_type = captured_piece.symbol().lower()
                if piece_type == 'p':
                    reward += 2  # Reward for capturing a pawn
                elif piece_type in ['n', 'b']:
                    reward += 7.5  # Reward for capturing a knight or bishop
                elif piece_type == 'r':
                    reward += 10  # Reward for capturing a rook
                elif piece_type == 'q':
                    reward += 15  # Reward for capturing a queen
        else:
            reward += 0.01  # Small reward for making a regular move
        return reward

    def is_game_over(self):
        # Checks if the game is over (checkmate, stalemate, or draw)
        return self.board.is_game_over()

    def render(self):
        # Renders the current state of the chessboard on the screen
        self.draw_board()
        self.draw_pieces()
        self.display_reward()
        p.display.flip()
        self.clock.tick(30)

    def draw_board(self):
        # Draws the chessboard with alternating colors for squares
        for row in range(8):
            for col in range(8):
                color = self.colors[(row + col) % 2]
                p.draw.rect(self.screen, color, p.Rect(col * self.SQUARE_SIZE, row * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE))

    def draw_pieces(self):
        # Draws the chess pieces on the board in their current positions
        for row in range(8):
            for col in range(8):
                piece = self.board.piece_at(row * 8 + col)
                if piece:
                    self.screen.blit(self.piece_images[piece.symbol().lower()], p.Rect(col * self.SQUARE_SIZE, row * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE))

    def display_reward(self):
        # Displays the current reward on the screen
        reward_text = self.font.render(f'Reward: {self.current_reward:.2f}', True, p.Color("black"))
        self.screen.blit(reward_text, (10, 10))
