using datasetGen;
using ShellProgressBar;
using System.Diagnostics;

public class Program
{
	public const string FileName = "exported.csv";
	public const int GoalAmount = 2_000_000;

	public const int Precision = 100_000;

	public static void Main(string[ ] args)
	{
		var boardGenerator = new ConnectFourStateGenerator();
		var progressBarSettings = new ProgressBarOptions
		{
			ProgressCharacter = '█',
			ForegroundColor = ConsoleColor.Yellow,
			ForegroundColorDone = ConsoleColor.Green,
			DisplayTimeInRealTime = false,
			CollapseWhenFinished = false
		};

		Console.CancelKeyPress += (sender, e) =>
		{
			e.Cancel = true;
			boardGenerator.CancellationRequested = true;
		};

		var progressBar = new ProgressBar(Precision, "Generating...", progressBarSettings);

		while (boardGenerator.GeneratedSets.Count <= GoalAmount && !boardGenerator.CancellationRequested /*boardGenerator.TryGenerate(goalAmount)*/)
		{
			boardGenerator.KickOffBoardStateGeneration();
			progressBar.Tick((int)( (double)boardGenerator.GeneratedSets.Count / GoalAmount * Precision ));
		}
		progressBar.Tick(Precision);
		progressBar.WriteLine("Finished generating");
		SaveData(boardGenerator, new(Precision, "Saving...", progressBarSettings), Precision);
	}

	public static void SaveData(ConnectFourStateGenerator generator, ProgressBar progressBar, int precision)
	{
		try
		{
			using var stream = File.OpenWrite(FileName);
			using var writer = new StreamWriter(stream);
			CsvHelper.WriteToCsv(writer, generator.GeneratedSets, progressBar, generator.TileCount, precision);
		} catch (Exception ex)
		{
			Console.WriteLine(ex);
		}
	}
}