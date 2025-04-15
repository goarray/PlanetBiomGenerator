unit AddBiomesFromFile;

//Overwrites Biomes in selected PNDT with biomes from Biomes.csv
//Tip, use the DumpFormIDEDID script to export selected biomes (or the IDs of any records)

var
  FileName: string;
  Lines: TStringList;
  BiomeIDs: TStringList;
  Biomes: IInterface;
  i, BiomeIndex, ChanceValue: Integer;
  PPBD, BiomeEntry: IInterface;
  HexBiomeID: string;
  Parts: TStringList;
  
const
  MAX_BIOMES = 7;
  
procedure ShuffleList(List: TStringList);
var
  i, j: Integer;
  temp: string;
begin
  Randomize;
  for i := List.Count - 1 downto 1 do
  begin
    j := Random(i + 1);
    temp := List[i];
    List[i] := List[j];
    List[j] := temp;
  end;
end;

function CleanBiomeID(BiomeID: string): string;
begin
  // Remove any non-numeric characters (e.g., parentheses, spaces)
  Result := StringReplace(BiomeID, ')', '', [rfReplaceAll]);
  Result := StringReplace(Result, '(', '', [rfReplaceAll]);
  Result := Trim(Result); // Clean up any surrounding spaces
end;

function ConvertToHex(BiomeID: string): string;
var
  DecimalID: Integer;
  CleanedID: string;
begin
  try
    CleanedID := CleanBiomeID(BiomeID);

    if Copy(CleanedID, 1, 2) = '00' then
      DecimalID := StrToInt('$' + CleanedID)
    else
      DecimalID := StrToInt(CleanedID);

    Result := Format('%.8X', [DecimalID]);
  except
    on E: Exception do
    begin
      AddMessage('Error converting BiomeID: "' + CleanedID + '" is invalid.');
      Result := '';
    end;
  end;
end;

function AddOrModifyBiome(Biomes: IInterface; BiomeID: string; ChanceValue: Integer; Index: Integer): Boolean;
var
  BiomeEntry: IInterface;
  HexBiomeID: Cardinal;
begin
  Result := False;

  // Trim and convert to FormID
  BiomeID := Trim(BiomeID);
  try
    HexBiomeID := StrToInt('$' + BiomeID);
  except
    on E: Exception do
    begin
      AddMessage('Error converting BiomeID: "' + BiomeID + '" is invalid.');
      Exit;
    end;
  end;

  // Check if index exists, otherwise create
  if Index < ElementCount(Biomes) then
    BiomeEntry := ElementByIndex(Biomes, Index)
  else
    BiomeEntry := ElementAssign(Biomes, HighInteger, nil, False);

  if not Assigned(BiomeEntry) then
  begin
    AddMessage('Error: Failed to get or create Biome entry at index ' + IntToStr(Index));
    Exit;
  end;

  // Set Biome FormID (element 0) and Chance (element 1)
  SetEditValue(ElementByIndex(BiomeEntry, 0), IntToHex(HexBiomeID, 8));
  SetEditValue(ElementByIndex(BiomeEntry, 1), IntToStr(ChanceValue));

  AddMessage('Set Biome ' + IntToHex(HexBiomeID, 8) + ' with chance ' + IntToStr(ChanceValue) + ' at index ' + IntToStr(Index));

  Result := True;
end;

function Process(e: IInterface): Integer;
var
  FileName: string;
  Lines, BiomeIDs, Parts: TStringList;
  i, BiomeIndex, ChanceValue: Integer;
  Biomes: IInterface;
begin
  Result := 0;
  FileName := ProgramPath + 'biomes.csv';

  Lines := TStringList.Create;
  BiomeIDs := TStringList.Create;

  try
    Lines.LoadFromFile(FileName);
    AddMessage('Loaded ' + IntToStr(Lines.Count) + ' lines from ' + FileName);

    if Lines.Count < 2 then
    begin
      AddMessage('Error: File does not contain enough lines.');
      Exit;
    end;

    // Loop over each line, skipping header
    for i := 1 to Lines.Count - 1 do
    begin
      Parts := TStringList.Create;
      try
        Parts.Delimiter := ',';  // Comma-separated
        Parts.DelimitedText := Lines[i];

        if Parts.Count >= 1 then
          BiomeIDs.Add(Parts[0]);  // FormID only
      finally
        Parts.Free;
      end;
    end;

    AddMessage('Parsed ' + IntToStr(BiomeIDs.Count) + ' biome IDs from CSV.');

    // Locate the Biomes list in the PNDT record
    Biomes := ElementByPath(e, 'Biomes');
    if not Assigned(Biomes) then
    begin
      AddMessage('Error: Biomes path not found in PNDT record.');
      Exit;
    end;

    AddMessage('Number of Biomes entries: ' + IntToStr(ElementCount(Biomes)));

    // Remove extra biomes beyond MAX_BIOMES
    while ElementCount(Biomes) > MAX_BIOMES do
      Remove(ElementByIndex(Biomes, ElementCount(Biomes) - 1));

    // Shuffle biome list to get a random selection
    ShuffleList(BiomeIDs);

    // Apply only the first MAX_BIOMES shuffled biome IDs
    for BiomeIndex := 0 to Min(BiomeIDs.Count, MAX_BIOMES) - 1 do
    begin
      ChanceValue := Random(100) + 1; // Random chance from 1 to 100
      if not AddOrModifyBiome(Biomes, BiomeIDs[BiomeIndex], ChanceValue, BiomeIndex) then
        AddMessage('Error: Failed to add or modify Biome entry.');
    end;

  finally
    Lines.Free;
    BiomeIDs.Free;
  end;
end;


end.
