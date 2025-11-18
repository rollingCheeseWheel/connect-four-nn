using ShellProgressBar;

namespace datasetGen;

public class CsvHelper
{
	public static void WriteToCsv(StreamWriter writer, IEnumerable<(Board, Player, int)> data, ProgressBar progressBar, int tileCount, int precision = 1000, int chunkSize = 1000)
	{
		var headers = Enumerable.Range(0, tileCount)
			.Select(i => i.ToString())
			.Concat([ "player", "bestcol" ])
			.ToArray();

		writer.WriteLine(string.Join(",", headers));

		double elapsedSize = 0;
		foreach (var chunk in data.Chunk(chunkSize))
		{
			foreach (var entry in chunk)
			{
				var (board, player, bestcol) = entry;
				var tempLine = board.FlattenToInt()
					.Concat([ (int)player, bestcol ])
					.ToList();
				writer.WriteLine(string.Join(",", tempLine));
			}
			elapsedSize += chunk.Count();
			progressBar.Tick((int)( elapsedSize / data.Count() * precision ));
		}
		progressBar.Tick(precision);
		progressBar.WriteLine("Finished writing");
	}
}
