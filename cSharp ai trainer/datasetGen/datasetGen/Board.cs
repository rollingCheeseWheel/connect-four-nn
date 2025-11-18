using System.Data;
using System.Diagnostics.CodeAnalysis;

namespace datasetGen;

public interface IConnectFourStateGenerator
{
	public bool InitBoardStateGeneration();
	public bool InitBoardStateGeneration(out IEnumerable<Board> generatedBoards);
}

public class ConnectFourStateGenerator(Width width, Height height) : IConnectFourStateGenerator
{
	public readonly Width Width = width;
	public readonly Height Height = height;
	public readonly int TileCount = width * height;

	public int[ ] MoveHistory { get; } = new int[ width * height ];
	public HashSet<Board> GeneratedSets = new();

	public bool InitBoardStateGeneration() => InitBoardStateGeneration(out var _);
	public bool InitBoardStateGeneration(out IEnumerable<Board> generatedBoards)
	{
		var board = new Board(Width, Height);
		var rowCounter = board.GetRowCounter();

		throw new NotImplementedException();
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
	/// <summary>
	/// The array has the following structure: [bottom-most, ..., top-most]
	/// </summary>
	/// <param name="column"></param>
	/// <returns></returns>
	public readonly Player[ ] GetColumn(int column)
	{
		var localCopy = this;
		return Enumerable.Range(0, Values.GetLength(0)).Select(i => localCopy[ column, i ]).ToArray();
	}
	/// <summary>
	/// The arra has the following structure: [left-most, ..., right-most]
	/// </summary>
	/// <param name="row"></param>
	/// <returns></returns>
	public readonly Player[ ] GetRow(int row)
	{
		var localCopy = this;
		return Enumerable.Range(0, Values.GetLength(1)).Select(i => localCopy[ i, row ]).ToArray();
	}

	public readonly void Set(int column, int row, Player player) => Values[ column, row ] = player;
	public bool TryMove(int column, Player player)
	{
		for (var row = 0; row < Height; row++) // iterate bottom up
		{
			if (this[ column, row ] != Player.Empty)
			{
				this[ column, row ] = player;
				return true;
			}
		}
		return false;
	}

	public int[ ] GetRowCounter()
	{
		var localCopy = this;
		return new int[ Width ].Select(x => (int)localCopy.Height).ToArray();
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

	public static int operator +(Operable first, Operable second) => first with { Value = first.Value + second.Value };
	public static int operator -(Operable first, Operable second) => first with { Value = first.Value - second.Value };
	public static int operator *(Operable first, Operable second) => first with { Value = first.Value * second.Value };
	public static int operator /(Operable first, Operable second) => first with { Value = first.Value / second.Value };
}

public record Height(int Value) : Operable(Value);
public record Width(int Value) : Operable(Value);