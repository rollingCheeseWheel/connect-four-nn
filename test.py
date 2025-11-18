#	import pandas, numpy
#	from model import *
#	
#	mod: Model = Model.load("connectFour.pickle")
#	df = pandas.read_csv("GC4BStates.csv")
#	
#	input = df.loc[1].values[1:-1]
#	target = df.loc[1].values[-1]
#	
#	pred = mod.forward(input)
#	maxPred = numpy.max(pred)
#	index = [i for i in range(pred.__len__()) if pred[i] == maxPred]
#	
#	print(f"prediction: {pred}", f"index: {index}", f"target: {target}", sep="\n")

#	import random
#	champArray = "Lux\r\nNautilus\r\nElise\r\nNeeko\r\nIvern\r\nTristana\r\nTwisted Fate\r\nKindred\r\nRyze\r\nSejuani\r\nYou #	Choose\r\nViktor\r\nCaitlyn\r\nTaric\r\nCamille\r\nQuinn\r\nKarma\r\nBard\r\nAkali\r\nVolibear\r\nVel'Koz\r\nDarius\r\nDiana\r\n#	Nasus\r\nAmumu\r\nKatarina\r\nKarthus\r\nThresh\r\nBraum\r\nKayn\r\nTahm #	Kench\r\nXayah\r\nSoraka\r\nLissandra\r\nGaren\r\nKled\r\nLee #	Sin\r\nUrgot\r\nZoe\r\nJinx\r\nSwain\r\nWarwick\r\nZed\r\nVladimir\r\nIllaoi\r\nRakan\r\nHecarim\r\nMaster #	Yi\r\nKog'Maw\r\nRiven\r\nEvelynn\r\nAzir\r\nOrnn\r\nRek'Sai\r\nOlaf\r\nLeBlanc\r\nJanna\r\nMorgana\r\nGnar\r\nPyke\r\nSyndra\r\#	nYou Choose\r\nNunu & Willump\r\nAnnie\r\nVi\r\nEzreal\r\nAshe\r\nShen\r\nTalon\r\nLucian\r\nMiss #	Fortune\r\nHeimerdinger\r\nJhin\r\nFiddlesticks\r\nMalphite\r\nKai'Sa\r\nFizz\r\nKennen\r\nTaliyah\r\nZyra\r\nTryndamere\r\nLeon#	a\r\nSylas\r\nVayne\r\nCorki\r\nCho'Gath\r\nRenekton\r\nGangplank\r\nAurelion #	Sol\r\nPantheon\r\nTrundle\r\nZac\r\nKha'Zix\r\nZiggs\r\nUdyr\r\nRammus\r\nNidalee\r\nAhri\r\nDr. #	Mundo\r\nLulu\r\nIrelia\r\nSona\r\nXin #	Zhao\r\nSkarner\r\nMaokai\r\nGragas\r\nBrand\r\nSivir\r\nJax\r\nFiora\r\nEkko\r\nZilean\r\nPoppy\r\nKassadin\r\nJarvan #	IV\r\nGraves\r\nTwitch\r\nMordekaiser\r\nYorick\r\nYasuo\r\nTeemo\r\nKalista\r\nShyvana\r\nAatrox\r\nXerath\r\nSion\r\nVarus\r\n#	Kayle\r\nWukong\r\nNami\r\nMalzahar\r\nAlistar\r\nShaco\r\nJayce\r\nDraven\r\nCassiopeia\r\nVeigar\r\nRumble\r\nNocturne\r\nSing#	ed\r\nBlitzcrank\r\nOrianna\r\nAnivia\r\nGalio\r\nRengar".split("\r\n")
#	while True:
#		random.shuffle(champArray)
#		for champ in champArray:
#			print(champ, end="")
#			input()