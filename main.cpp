#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <random>
#include <map>
#include <memory>
#include "matplotlibcpp.h"
#include <tuple>
#include <algorithm>

using namespace std;

namespace fs = std::filesystem;
namespace plt = matplotlibcpp;

torch::Tensor get_accuracy_score(torch::Tensor predictions, torch::Tensor targets);

class PlantDiseaseDataset : public torch::data::Dataset<PlantDiseaseDataset>
{
private:
    vector<tuple<string, string>> images_labels;
    vector<string> labels;
    pair<int, int> image_shape;
    string channels;
    bool use_augmentations;
    mt19937 gen;
    uniform_real_distribution<> dis;

public:
    PlantDiseaseDataset(string &path)
    {
        image_shape = {256, 256};
        channels = "RGB";
        use_augmentations = false;

        if (fs::exists(path))
        {
            for (const auto &entry : fs::directory_iterator(path))
            {
                if (entry.is_directory())
                {
                    string label = entry.path().filename().string();
                    labels.push_back(label);
                    int count = 0;
                    for (const auto &file : fs::directory_iterator(entry.path()))
                    {
                        string filename = file.path().string();
                        if (filename.ends_with(".jpg") || filename.ends_with(".png"))
                        {
                            images_labels.push_back({filename, label});
                            count++;
                        }
                    }

                    cout << "Load " << count << " images with label " << label << endl;
                }
            }
        }
    }

    void apply_augmentations(cv::Mat &image)
    {
        if (dis(gen) < 0.5)
        {
            cv::flip(image, image, 1);
        }

        if (dis(gen) < 0.5)
        {
            cv::flip(image, image, 0);
        }
    }

    torch::optional<size_t> size() const override
    {
        return images_labels.size();
    }

    torch::Tensor load_image(const string &path, int width, int height, const string &channels = "RGB")
    {
        cv::Mat image = cv::imread(path);

        if (image.empty())
        {
            throw std::runtime_error("Failed to load image: " + path);
        }

        cv::resize(image, image, cv::Size(width, height));

        if (use_augmentations)
        {
            apply_augmentations(image);
        }

        if (channels == "RGB")
        {
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        }
        else if (channels == "L" || channels == "GRAY")
        {
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        }

        image.convertTo(image, CV_32FC3, 1.0 / 255.0);

        torch::Tensor tensor = torch::from_blob(image.data, {image.rows, image.cols, image.channels()}, torch::kFloat32).clone();

        tensor = tensor.permute({2, 0, 1});

        return tensor;
    }

    torch::data::Example<> get(size_t index) override
    {
        tuple<string, string> image_data = images_labels[index];

        string path = get<0>(image_data);
        string label = get<1>(image_data);

        int width = get<0>(image_shape);
        int height = get<1>(image_shape);

        torch::Tensor image = load_image(path, width, height);

        int label_index = 0;
        for (; label_index < labels.size(); label_index++)
        {
            if (labels[label_index] == label)
            {
                break;
            }
        }

        torch::Tensor label_tensor = torch::tensor(label_index, torch::kInt64);

        return {image, label_tensor};
    }
};

class PlantDiseaseModel : public torch::nn::Module
{
private:
    torch::jit::script::Module model;
    string model_path = "./resnet34_pretrained.pt";

public:
    PlantDiseaseModel() : torch::nn::Module()
    {
        if (fs::exists(model_path))
        {
            model = torch::jit::load(model_path);

            cout << "Model loaded!" << endl;
        }
        else
        {
            cout << "Model file was not found..." << endl;
        }
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return model.forward({x}).toTensor();
    }

    void to(torch::Device device)
    {
        model.to(device);
    }
};

class Trainer
{
private:
    PlantDiseaseModel model;
    torch::nn::CrossEntropyLoss criterion;
    torch::optim::Optimizer *optimizer = nullptr;
    void *logger;
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    double best_validation_loss;
    std::map<std::string, std::vector<double>> history;

    void log(std::map<std::string, double> logs)
    {
        for (const auto &[key, value] : logs)
        {
            if (history.find(key) == history.end())
            {
                history[key] = std::vector<double>();
            }
            history[key].push_back(value);
        }
    }

public:
    Trainer(PlantDiseaseModel train_model,
            unique_ptr<torch::optim::Optimizer> &train_optim,
            void *train_logger = nullptr)
    {
        model = std::move(train_model);
        criterion = torch::nn::CrossEntropyLoss();
        optimizer = train_optim.get();
        logger = train_logger;
        best_validation_loss = 0.0;
        history = std::map<std::string, std::vector<double>>();
    }

    template <typename DataLoader>
    pair<double, double> evaluate(DataLoader &loader)
    {
        double loss = 0.0;
        double score = 0.0;
        int length = 0;

        model.to(device);
        torch::NoGradGuard no_grad;

        model.eval();

        for (auto &batch : loader)
        {
            std::vector<torch::Tensor> images_vec, labels_vec;
            for (const auto &example : batch)
            {
                images_vec.push_back(example.data);
                labels_vec.push_back(example.target);
            }

            auto images = torch::stack(images_vec).to(torch::kFloat32).to(device);
            auto labels = torch::stack(labels_vec).to(torch::kLong).to(torch::kCPU);

            auto probabilities = model.forward(images).to(torch::kFloat32).to(torch::kCPU);
            auto predictions = torch::argmax(probabilities, 1).detach();

            auto batch_loss = criterion(probabilities, labels);
            loss += batch_loss.template item<double>();

            auto batch_score = ::get_accuracy_score(predictions, labels).template item<double>();

            score += batch_score;

            length++;
        }

        loss /= length;
        score /= length;

        return make_pair(loss, score);
    }

    template <typename DataLoader>
    void fit(DataLoader &train_loader,
             DataLoader &validation_loader,
             int epochs = 10)
    {
        model.to(device);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double epoch_loss = 0.0;
            double epoch_score = 0.0;
            int train_length = 0;

            cout << "Epoch [" << (epoch + 1) << "/" << epochs << "]: " << std::endl;

            for (auto &batch : train_loader)
            {
                optimizer->zero_grad();
                model.train();

                std::vector<torch::Tensor> images_vec, labels_vec;
                for (const auto &example : batch)
                {
                    images_vec.push_back(example.data);
                    labels_vec.push_back(example.target);
                }

                auto images = torch::stack(images_vec).to(torch::kFloat32).to(device);
                auto labels = torch::stack(labels_vec).to(torch::kLong).to(torch::kCPU);

                auto probabilities = model.forward(images).to(torch::kFloat32).to(torch::kCPU);
                auto predictions = torch::argmax(probabilities, 1).detach();

                auto batch_loss = criterion(probabilities, labels);
                epoch_loss += batch_loss.template item<double>();

                auto batch_score = ::get_accuracy_score(predictions, labels).template item<double>();
                epoch_score += batch_score;

                batch_loss.backward();
                optimizer->step();

                train_length++;
            }

            epoch_loss /= train_length;
            epoch_score /= train_length;

            std::map<std::string, double> train_logs = {
                {"train_losses", epoch_loss},
                {"train_scores", epoch_score}};
            log(train_logs);

            cout << "Epoch [" << (epoch + 1) << "/" << epochs << "]: Loss: " << epoch_loss << " | Metric: " << epoch_score << std::endl;

            if (true)
            { // Always run validation if loader is provided
                auto [validation_loss, validation_score] = evaluate(validation_loader);

                std::map<std::string, double> val_logs = {
                    {"validation_losses", validation_loss},
                    {"validation_scores", validation_score}};
                log(val_logs);

                std::cout << "Validation Epoch [" << (epoch + 1) << "/" << epochs << "]: Loss: " << validation_loss << " | Metric: " << validation_score << std::endl;
            }
        }
    }
};

struct TrainConfig
{
public:
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    int epochs;
    int seed;
    int width;
    int height;
    int channels;
    int num_workers;
    int batch_size;

    std::string optimizer_type;
    double learning_rate;
    double weight_decay;

    std::string scheduler_type;
    int patience;
    std::string mode;
    double factor;

    TrainConfig()
    {
        epochs = 5;
        seed = 2021;
        width = 128;
        height = 128;
        channels = 3;
        num_workers = 0;
        batch_size = 32;

        optimizer_type = "AdamW";
        learning_rate = 0.001;
        weight_decay = 0.01;

        scheduler_type = "ReduceLROnPlateau";
        patience = 2;
        mode = "min";
        factor = 0.1;
    }
};

struct CustomCollate : public torch::data::transforms::Collation<torch::data::Example<>>
{
    torch::data::Example<torch::Tensor> apply_batch(
        std::vector<torch::data::Example<>> examples) override
    {
        std::vector<torch::Tensor> all_images;
        std::vector<int64_t> all_labels;

        for (auto &example : examples)
        {
            all_images.push_back(example.data);
            all_labels.push_back(example.target.item<int64_t>());
        }

        torch::Tensor images = torch::stack(all_images);
        torch::Tensor labels = torch::tensor(all_labels, torch::kLong);

        return {images, labels};
    }
};

void seed_everything()
{
    unsigned int seed = std::random_device{}();
    torch::manual_seed(seed);
    torch::cuda::manual_seed(seed);
    cout << "Seed was set to " << seed << endl;
}

unique_ptr<torch::optim::Optimizer> get_optimizer(
    const torch::nn::Module &model,
    string name = "SGD",
    map<string, double> params = {})
{
    double lr = 0.01;
    double wd = 0.0;

    if (params.find("lr") != params.end())
    {
        lr = params["lr"];
    }
    if (params.find("weight_decay") != params.end())
    {
        wd = params["weight_decay"];
    }

    if (name == "SGD")
    {
        return make_unique<torch::optim::SGD>(
            model.parameters(),
            torch::optim::SGDOptions(lr).weight_decay(wd));
    }
    else if (name == "Adam")
    {
        return make_unique<torch::optim::Adam>(
            model.parameters(),
            torch::optim::AdamOptions(lr).weight_decay(wd));
    }
    else if (name == "AdamW")
    {
        return make_unique<torch::optim::AdamW>(
            model.parameters(),
            torch::optim::AdamWOptions(lr).weight_decay(wd));
    }

    return make_unique<torch::optim::RMSprop>(
        model.parameters(),
        torch::optim::RMSpropOptions(lr).weight_decay(wd));
}

torch::Tensor get_accuracy_score(torch::Tensor predictions, torch::Tensor targets)
{
    torch::Tensor amount = (predictions == targets).sum();
    torch::Tensor accuracy = amount / targets.size(0);

    return accuracy;
}

int main()
{
    seed_everything();

    string folder_base_path = "./"; // /Users/petrugrigor/Documents

    map<string, string> config = {
        {"train_path", folder_base_path + "/Data/Train"},
        {"test_path", folder_base_path + "/Data/Test"},
        {"validation_path", folder_base_path + "/Data/Validation"}};

    PlantDiseaseDataset train_dataset = PlantDiseaseDataset(config["train_path"]);
    PlantDiseaseDataset validation_dataset = PlantDiseaseDataset(config["validation_path"]);
    PlantDiseaseDataset test_dataset = PlantDiseaseDataset(config["test_path"]);

    TrainConfig train_config = TrainConfig();

    cout << "Train dataset size " << train_dataset.size() << endl;
    cout << "Test dataset size " << test_dataset.size() << endl;

    // auto train_loader = torch::data::make_data_loader(
    //     std::move(train_dataset),
    //     torch::data::DataLoaderOptions()
    //         .batch_size(train_config.batch_size)
    //         .workers(train_config.num_workers)
    //         .enforce_ordering(false));

    // auto test_loader = torch::data::make_data_loader(
    //     std::move(test_dataset),
    //     torch::data::DataLoaderOptions()
    //         .batch_size(train_config.batch_size)
    //         .workers(train_config.num_workers));

    // auto val_loader = torch::data::make_data_loader(
    //     std::move(validation_dataset),
    //     torch::data::DataLoaderOptions()
    //         .batch_size(train_config.batch_size)
    //         .workers(train_config.num_workers));

    // PlantDiseaseModel model = PlantDiseaseModel();

    // unique_ptr<torch::optim::Optimizer> optimizer = get_optimizer(model, train_config.optimizer_type);

    // Trainer trainer = Trainer(model, optimizer);

    // trainer.fit(*train_loader, *val_loader, train_config.epochs);

    return 0;
}
