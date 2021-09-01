package org.pp.data;

import lombok.SneakyThrows;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.stream.Collectors;

import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher;
import org.nd4j.linalg.factory.Nd4j;

public class StockCSVDataSetFetcher implements DataSetFetcher {

    private int ReaderCursor = 0;
    private int FetcherCursor = 0;
    private final int inputColumns;
    private final int totalOutcomes;
    private final int totalExamples;

    private ArrayList<String> allLines;

    private File csvFile;

    private ArrayList<DataSet> fetch;

    private int lineCount;

    public StockCSVDataSetFetcher(File csvFile){
        this(csvFile, 100, 10);
    }

    @SneakyThrows
    public StockCSVDataSetFetcher(File csvFile, int inputColumns, int totalOutcomes){
        this.csvFile = csvFile;
        this.inputColumns = inputColumns;
        this.totalOutcomes = totalOutcomes;
        this.fetch = new ArrayList<>();

        this.lineCount = getLineCount();
        this.totalExamples = this.calcTotalExamples();

        this.allLines = Files.lines(Paths.get(csvFile.getPath()))
                .skip(1) // skipping csv headers
                .collect(Collectors.toCollection(ArrayList::new));
    }

    @SneakyThrows
    private int getLineCount(){
        BufferedReader reader = new BufferedReader(new FileReader(csvFile));
        int lines = 0;
        while (reader.readLine() != null) lines++;
        reader.close();

        return lines - 1;
    }

    @Override
    public boolean hasMore() {
        return ReaderCursor < this.totalExamples();
    }

    @Override
    public DataSet next() {
        DataSet dataSet = fetch.get(FetcherCursor);

        FetcherCursor++;
        return dataSet;
    }


    @SneakyThrows
    @Override
    public void fetch(int numExamples) {
        FetcherCursor = 0;

        fetch.clear();

        for (int i = 0; i < numExamples; i++) {
            INDArray input = Nd4j.create(1, inputColumns, 1);
            INDArray output = Nd4j.create(1, totalOutcomes, 1);

            //2013-02-08,45.07,45.35,45.0,45.08,1824755,A
            for (int j = 0; j < inputColumns; j++){
                double val = Double.parseDouble(allLines.get(ReaderCursor + j + i).split(",")[4]);

                input.putScalar(0, j,0, val);
            }

            for (int z = 0; z < totalOutcomes; z++){
                double val = Double.parseDouble(allLines.get(ReaderCursor + inputColumns + z + i).split(",")[4]);

                output.putScalar(0, z, 0, val);
            }

            fetch.add(new DataSet(input, output));

            //pop data
            ReaderCursor++;
        }
    }

    @Override
    public int totalOutcomes() {
        return totalOutcomes;
    }

    @Override
    public int inputColumns() {
        return inputColumns;
    }

    @Override
    public int totalExamples() {
        return this.totalExamples;
    }

    private int calcTotalExamples() {
        return lineCount - (inputColumns + totalOutcomes + 64);
    }

    @Override
    public void reset() {
        ReaderCursor = 0;
    }

    @Override
    public int cursor() {
        return FetcherCursor;
    }
}
