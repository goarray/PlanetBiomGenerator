unit ExportPlanetBiomes;

var
  sl: TStringList;
  pluginName: string;
  pluginHeaderWritten: Boolean;

function Process(e: IInterface): Integer;
var
  planetName, biomeFormID, biomeEDID: string;
  biomes, biomeEntry, biomeRef: IInterface;
  i: Integer;
begin
  Result := 0;

  // Get planet name (EditorID)
  planetName := GetElementEditValues(e, 'Body\ANAM');

  // Get Biomes array
  biomes := ElementByPath(e, 'Biomes');

  // Get plugin name once
  if pluginName = '' then
    pluginName := LowerCase(GetFileName(GetFile(e)));

  // Write plugin name and CSV header once
  if not pluginHeaderWritten then begin
    sl.Add(pluginName);  // First line = plugin name
    sl.Add('PlanetName,BIOM_FormID,BIOM_EditorID');  // Second line = CSV header
    pluginHeaderWritten := True;
  end;

  // If no biomes, output dummy entry
  if not Assigned(biomes) or (ElementCount(biomes) = 0) then begin
    sl.Add(Format('%s,%s,%s', [planetName, '0017B861', '']));
    Exit;
  end;

  // Loop through biome entries
  for i := 0 to ElementCount(biomes) - 1 do begin
    biomeEntry := ElementByIndex(biomes, i);
    biomeRef := LinksTo(ElementByIndex(biomeEntry, 0)); // Get actual BIOM record

    if Assigned(biomeRef) then begin
      biomeFormID := IntToHex(FormID(biomeRef), 8);
      biomeEDID := GetElementEditValues(biomeRef, 'EDID');
    end else begin
      biomeFormID := '00000000';
      biomeEDID := '';
    end;

    sl.Add(Format('%s,%s,%s', [planetName, biomeFormID, biomeEDID]));
  end;
end;

function Initialize: Integer;
begin
  sl := TStringList.Create;
  pluginHeaderWritten := False;
end;

function Finalize: Integer;
var
  outputDir, outputPath: string;
begin
  outputDir := ProgramPath + '\PlanetBiomes\xEditOutput\';
  ForceDirectories(outputDir);
  outputPath := outputDir + 'PlanetBiomes.csv';

  sl.SaveToFile(outputPath);
  sl.Free;
end;

end.
