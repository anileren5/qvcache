/*
 * YcsbSeqGenerator.
 * Author: Sukjoon Oh (sjoon@kaist.ac.kr)
 */

#include <cstdint>
#include <algorithm>
#include <unordered_map>

#include <fstream>

#include "utils/ak_sequence_gen.hh"

YcsbSeqGenerator::YcsbSeqGenerator() noexcept
{
    resetGenerator();
}

std::uint64_t 
YcsbSeqGenerator::generateNextKey() noexcept
{
    std::uint64_t keyNumber = key_generator_->Next();
    return keyNumber;
}

std::uint64_t 
YcsbSeqGenerator::chooseNextKey() noexcept
{
    std::uint64_t keyNumber = 0;
    if (key_chooser_.get() != nullptr)
    {   
        do
        {
            keyNumber = key_chooser_->Next();
        } 
        while (keyNumber > insert_key_sequence_->Last());
    }
    return keyNumber;
}

ycsbc::Operation
YcsbSeqGenerator::chooseNextOp() noexcept
{
    ycsbc::Operation op = ycsbc::Operation::INSERT;
    if (op_chooser_.get() != nullptr)
    {
        op = op_chooser_->Next();
    }

    return op;
}

bool
YcsbSeqGenerator::setGenerator(
    size_t recordCount, 
    std::string distType,
    double insertRatio,
    double updateRatio,
    double readRatio
) noexcept
{
    insert_key_sequence_->Set(recordCount);

    // Make the distType lowercase.
    for (char &c : distType) c = c | ' ';

    if (distType == "uniform")
        key_chooser_.reset(new ycsbc::UniformGenerator(0, recordCount - 1));

    else if (distType == "zipfian")
        key_chooser_.reset(new ycsbc::ScrambledZipfianGenerator(recordCount));

    else if (distType == "latest")
        key_chooser_.reset(new ycsbc::SkewedLatestGenerator(*insert_key_sequence_));

    else
    {
        key_chooser_.reset();
        return false;
    }

    if (insertRatio > 0)
        op_chooser_->AddValue(ycsbc::Operation::INSERT, insertRatio);

    if (updateRatio > 0)
        op_chooser_->AddValue(ycsbc::Operation::UPDATE, updateRatio);

    if (readRatio > 0)
        op_chooser_->AddValue(ycsbc::Operation::READ, readRatio);

    return true;
}

/**
 * Resets the generator to its initial state.
 */
void 
YcsbSeqGenerator::resetGenerator() noexcept
{
    key_sequence_.clear();
    op_sequence_.clear();

    // Make default
    insert_key_sequence_.reset(new ycsbc::CounterGenerator(3));
    key_generator_.reset(new ycsbc::CounterGenerator(0));

    op_chooser_.reset(
        new ycsbc::DiscreteGenerator<ycsbc::Operation>()
    );

    key_chooser_.reset();
}

std::vector<std::uint64_t>&
YcsbSeqGenerator::generateKeySequence(size_t queryVectorNumber) noexcept
{
    for (size_t count = 0; count < queryVectorNumber; count++)
        key_sequence_.emplace_back(chooseNextKey());

    // This class just generates the sequence with the given distribution set.
    // Mapping of external vectors should be done by the caller.
    // Use retuned sequence ID to map the vectors.

    return key_sequence_;
}

size_t
YcsbSeqGenerator::checkUniqueIds(
    std::vector<std::pair<std::uint64_t, size_t>>& idsByFreq
) noexcept
{
    // Extract unique keys
    std::unordered_map<std::uint64_t, size_t> uniqueKeys;
    for (std::uint64_t& key: key_sequence_)
    {
        if (uniqueKeys.find(key) == uniqueKeys.end())
            uniqueKeys[key] = 1;
            
        else 
            uniqueKeys[key]++;
    }

    // Insert to the vector
    idsByFreq = std::vector<std::pair<std::uint64_t, size_t>>(
        uniqueKeys.begin(), uniqueKeys.end()
    );

    // Sort the vector by frequency (descending order)
    std::sort(
        idsByFreq.begin(),
        idsByFreq.end(),
        [](const std::pair<std::uint64_t, size_t>& a, const std::pair<std::uint64_t, size_t>& b) {
            return a.second > b.second;
        }
    );

    return uniqueKeys.size();
}


void
YcsbSeqGenerator::exportFrequency() noexcept
{

    exportFrequency("sequence-freqs.csv");
}

void
YcsbSeqGenerator::exportFrequency(const std::string& output_path) noexcept
{

    std::vector<std::pair<std::uint64_t, size_t>> uniqueKeys;
    checkUniqueIds(uniqueKeys);

    // 
    // Export to files to visualize in descending count order.
    std::fstream outputFile(output_path, std::ios::out);
    if (!outputFile)
        return;

    for (const auto& pair: uniqueKeys)
        outputFile << pair.first << "\t" << pair.second << std::endl;

    outputFile.close();
}

void
YcsbSeqGenerator::exportSequence() noexcept
{

    exportSequence("sequence.csv");
}

void
YcsbSeqGenerator::exportSequence(const std::string& output_path) noexcept
{
    std::fstream outputFile(output_path, std::ios::out);
    if (!outputFile)
        return;

    for (const auto& key: key_sequence_)
        outputFile << key << "\t" << chooseNextOp() << std::endl;

    outputFile.close();
}