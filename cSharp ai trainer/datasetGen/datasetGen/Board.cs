using System.Data;
using System.Diagnostics.CodeAnalysis;

namespace datasetGen;

public interface IConnectFourStateGenerator
{
	public bool KickOffBoardStateGeneration();
	public bool KickOffBoardStateGeneration(out ICollection<Board> generatedBoards);
}

public class ConnectFourStateGenerator : IConnectFourStateGenerator
{
	public readonly Width Width = new(7);
	public readonly Height Height = new(6);
	public int TileCount => Width * Height;

	private int failuresToGenerate = 0;

	public bool CancellationRequested = false;

	public List<int> MoveHistory { get; } = [ ];
	public HashSet<(Board, Player, int)> GeneratedSets = new(); // state, player to move, best column
	public readonly Random rng = new();

	public bool TryGenerate(double goalAmount, int tolerance = 1_000_000)
	{
		if (GeneratedSets.Count >= goalAmount){
			return false;
		}

		int before = GeneratedSets.Count;
		KickOffBoardStateGeneration();
		// If no new states were added, stop
		if (GeneratedSets.Count > before)
		{
			failuresToGenerate = 0;
			return true;
		} else
		{
			failuresToGenerate++;
			return failuresToGenerate > tolerance;
		}
	}

	public bool KickOffBoardStateGeneration() => KickOffBoardStateGeneration(out var _);
	public bool KickOffBoardStateGeneration(out ICollection<Board> generatedBoards)
	{
		generatedBoards = [ ];
		var board = new Board(Width, Height);
		var rowCounter = board.GetColumnTracker(rng);

		// try iterating the longest possible time
		for (var i = 0; i < TileCount; i++)
		{
			// choose a column the player will play in
			var nextMove = rowCounter.GetRandomAvailableColumn();
			if (nextMove is null)
			{
				Console.WriteLine($"ran out of available columns: {i}");
				return false;
			}

			var currentPlayer = i % 2 == 0 ? Player.Human : Player.AI;
			MoveHistory.Add(nextMove.Value);
			if (!board.TryMove(nextMove.Value, currentPlayer))
			{
				Console.WriteLine($"Unable to move the player to that position, player: {currentPlayer}");
				return false;
			}

			if (!board.CheckWin() || i != TileCount - 1) { continue; } // continue, since there is no win or the moves are exausted

			for (var j = i; j >= 0; j--)
			{
				if (!board.TryUnMove(MoveHistory[ j ])) { throw new Exception($"Unexpected outcome, column cannot be unmoved (column: {j}"); }

				var backwardsIterPlayer = j % 2 == 0 ? Player.Human : Player.AI;
				(Board, Player, int) temp = (board.Clone(), backwardsIterPlayer, board.Clone().GetBestMove(backwardsIterPlayer) ?? short.MinValue);

				GeneratedSets.Add(temp);
				generatedBoards.Add(board.Clone());
			}
			break;
		}

		return true;
	}
}

public struct Board
{
	public readonly Width Width;
	public readonly Height Height;
	/// <summary>
	/// (bottom-most, left-most), ... (top-most, right-most)
	/// </summary>
	public Player[ , ] Values { get; private set; }

	public Board(Width width, Height height)
	{
		Width = width;
		Height = height;
		Values = new Player[ Width, Height ];
		Values.Initialize();
	}

	public readonly Player this[ Width column, Height row ]
	{
		get => Values[ column, row ];
		set => Values[ column, row ] = value;
	}
	private readonly Player this[ int column, int row ]
	{
		get => Values[ column, row ];
		set => Values[ column, row ] = value;
	}

	public readonly void Clear() => Values.Initialize();

	public readonly Player Get(int column, int row) => Values[ column, row ];
	/// <summary>The array has the following structure: [bottom-most, ..., top-most]</summary>
	public readonly Player[] GetColumn(int column)
	{
		var localCopy = this;
		return Enumerable.Range(0, Height).Select(row => localCopy[column, row]).ToArray();
	}
	/// <summary>The array has the following structure: [left-most, ..., right-most]</summary>
	public readonly Player[] GetRow(int row)
	{
		var localCopy = this;
		return Enumerable.Range(0, Width).Select(col => localCopy[col, row]).ToArray();
	}

	public readonly void Set(int column, int row, Player player) => Values[ column, row ] = player;
	public bool TryMove(int column, Player player)
	{
		for (var row = 0; row < Height; row++) // iterate bottom up
		{
			if (this[ column, row ] == Player.Empty)
			{
				this[ column, row ] = player;
				return true;
			}
		}
		return false;
	}

	public bool TryUnMove(int column)
	{
		for (var row = Height - 1; row >= 0; row--)
		{
			if (this[ column, row ] != Player.Empty)
			{
				this[ column, row ] = Player.Empty;
				return true;
			}
		}
		return false;
	}

	public ColumnTracker GetColumnTracker(Random? rng = null)
	{
		var localCopy = this;
		// Count empty slots in each column
		return new(
			Enumerable.Range(0, Width)
				.Select(column => localCopy.GetColumn(column).Count(p => p == Player.Empty))
				.ToArray(),
			rng
		);
	}

	/// <summary>Creates a deep clone of the board</summary>
	public readonly Board Clone()
	{
		var clone = new Board(Width, Height);
		Buffer.BlockCopy(Values, 0, clone.Values, 0, Values.Length * sizeof(int));
		return clone;
	}

	public Player[ ] Flatten() => Values.Cast<Player>().ToArray();
	public int[ ] FlattenToInt() => Flatten().Cast<int>().ToArray();
	public string ToCsvString() => string.Join(", ", FlattenToInt());

	public readonly bool CheckWin()
	{
		// Directions: vertical, horizontal, diagonal up-right, diagonal up-left
		var directions = new (int deltaColumn, int deltaRow)[ ] { (0, 1), (1, 0), (1, 1), (1, -1) };

		var localCopy = this;
		bool InBounds(int column, int row) => column >= 0 && column < localCopy.Width && row >= 0 && row < localCopy.Height;

		for (var row = 0; row < Height; row++)
		{
			for (var column = 0; column < Width; column++)
			{
				var player = this[ column, row ];
				if (player == Player.Empty)
					continue;

				foreach (var (deltaColumn, deltaRow) in directions)
				{
					var connectedCount = 1;

					for (var step = 1; step < 4; step++)
					{
						var nextColumn = column + deltaColumn * step;
						var nextRow = row + deltaRow * step;

						if (InBounds(nextColumn, nextRow) && this[ nextColumn, nextRow ] == player)
							connectedCount++;
						else
							break;
					}

					if (connectedCount >= 4)
						return true;
				}
			}
		}

		return false;
	}

	public int? GetBestMove(Player player)
	{
		var validColumns = GetColumnTracker().GetOpenColumns();
		var bestScore = int.MinValue;
		int? bestColumn = null;
		foreach (var column in validColumns)
		{
			var tempBoard = Clone();
			if (!tempBoard.TryMove(column, player)) { throw new Exception($"could not drop piece in column {column}"); }
			var tempScore = tempBoard.ScoreBoard(player);
			if (tempScore > bestScore)
			{
				bestColumn = column;
				bestScore = tempScore;
			}
		}
		return bestColumn;
	}

	public int ScoreBoard(Player player)
	{
		const int WINDOW_LENGTH = 4;
		int width = Width;
		int height = Height;
		int score = 0;

		// Center column score (prefer occupying center)
		int centerCol = width / 2;
		int centerCount = 0;
		for (int r = 0; r < height; r++)
		{
			if (Values[ centerCol, r ] == player)
				centerCount++;
		}
		score += centerCount * 3;

		// Horizontal windows
		for (int r = 0; r < height; r++)
		{
			for (int c = 0; c <= width - WINDOW_LENGTH; c++)
			{
				score += EvaluateWindow(c, r, 1, 0, player);
			}
		}

		// Vertical windows
		for (int c = 0; c < width; c++)
		{
			for (int r = 0; r <= height - WINDOW_LENGTH; r++)
			{
				score += EvaluateWindow(c, r, 0, 1, player);
			}
		}

		// Positive slope diagonals (up-right)
		for (int r = 0; r <= height - WINDOW_LENGTH; r++)
		{
			for (int c = 0; c <= width - WINDOW_LENGTH; c++)
			{
				score += EvaluateWindow(c, r, 1, 1, player);
			}
		}

		// Negative slope diagonals (down-right relative to visual; using row decreasing)
		for (int r = WINDOW_LENGTH - 1; r < height; r++)
		{
			for (int c = 0; c <= width - WINDOW_LENGTH; c++)
			{
				score += EvaluateWindow(c, r, 1, -1, player);
			}
		}

		return score;
	}

	private int EvaluateWindow(int startCol, int startRow, int dc, int dr, Player player)
	{
		const int WINDOW_LENGTH = 4;
		int pieceCount = 0;
		int emptyCount = 0;
		int oppCount = 0;
		var opponent = player == Player.Human ? Player.AI : Player.Human;

		for (int i = 0; i < WINDOW_LENGTH; i++)
		{
			int c = startCol + dc * i;
			int r = startRow + dr * i;
			Player cell = Values[ c, r ];
			if (cell == player) pieceCount++;
			else if (cell == Player.Empty) emptyCount++;
			else if (cell == opponent) oppCount++;
		}

		int score = 0;

		// Match Python evaluate_window scoring
		if (pieceCount == 4)
			score += 100;
		else if (pieceCount == 3 && emptyCount == 1)
			score += 5;
		else if (pieceCount == 2 && emptyCount == 2)
			score += 2;

		if (oppCount == 3 && emptyCount == 1)
			score -= 4;

		return score;
	}

	public static bool operator ==(Board first, Board second)
	{
		if (first.Width != second.Width || first.Height != second.Height) return false;
		for (var column = 0; column < first.Width; column++)
		{
			for (var row = 0; row < first.Height; row++)
			{
				if (first[ column, row ] != second[ column, row ])
				{
					return false;
				}
			}
		}
		return true;
	}
	public static bool operator !=(Board first, Board second) => !( first == second );
	public static bool operator ==(Board? first, Board? second)
	{
		if (first is null || second is null) return false;
		return first.Value == second.Value;
	}
	public static bool operator !=(Board? first, Board? second) => !( first == second );
	public override bool Equals([NotNullWhen(true)] object? other) => other is Board asBoard && asBoard == this;
	public readonly override int GetHashCode() // ChatGPT generated, idk what it does
	{
		unchecked
		{
			int hash = Width.GetHashCode();
			hash = ( hash * 31 ) ^ Height.GetHashCode();

			for (int x = 0; x < Width; x++)
				for (int y = 0; y < Height; y++)
					hash = ( hash * 31 ) ^ Values[ x, y ].GetHashCode();

			return hash;
		}
	}
	public override string ToString()
	{
		var rows = new List<string>();
		for (int row = Height - 1; row >= 0; row--)
		{
			var cells = new List<string>();
			for (int col = 0; col < Width; col++)
			{
				cells.Add(((int)Values[col, row]).ToString());
			}
			rows.Add(string.Join(" ", cells));
		}
		return string.Join(Environment.NewLine, rows);
	}
}

public class ColumnTracker(int[ ] columns, Random? rng = null)
{
	public int[ ] _tracker { get; } = columns;
	private readonly Random? _rng = rng;

	public int? GetRandomAvailableColumn()
	{
		var possibleColumns = GetOpenColumns();
		if (possibleColumns.Count() == 0) { return null; }
		var chosenColumn = _rng?.GetItems(possibleColumns, 1).First();
		if (chosenColumn is null) { return null; }
		_tracker[ (int)chosenColumn ] = Math.Max(_tracker[ (int)chosenColumn ] - 1, 0);
		return chosenColumn;
	}

	public int[ ] GetOpenColumns()
	{
		return new int[ _tracker.Length ]
			.Select((t, i) => i)
			.Where(i => _tracker[ i ] > 0)
			.ToArray();
	}
}

public enum Player
{
	Empty,
	Human,
	AI,
}

public record class Operable(int Value)
{
	public static implicit operator int(Operable operable) => operable.Value;
	public static implicit operator Operable(int value) => new(value);

	public static int operator +(Operable first, Operable second) => first with { Value = first.Value + second.Value };
	public static int operator -(Operable first, Operable second) => first with { Value = first.Value - second.Value };
	public static int operator *(Operable first, Operable second) => first with { Value = first.Value * second.Value };
	public static int operator /(Operable first, Operable second) => first with { Value = first.Value / second.Value };
}

public record Height(int Value) : Operable(Value);
public record Width(int Value) : Operable(Value);