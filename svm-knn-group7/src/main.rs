use linfa::prelude::*;
use linfa_svm::Svm;
use linfa_logistic::MultiLogisticRegression;
use ndarray::{Array1, Array2, Axis};
use plotters::prelude::*;
use csv::ReaderBuilder;
use std::collections::HashMap;
use std::fs;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::error::Error;
//use qmetaobject;
//use std::fs;

//#[derive(QObject, Default)]
//struct RustModel {
//    base: qt_base_class!(trait QObject),
//    result: qt_property!(QString; NOTIFY result_changed),
//    result_changed: qt_signal!(),
//    run: qt_method!(fn run(&mut self)), 
//}

//impl RustModel {
//    fn run(&mut self) {
//        let (features, labels) = load_data("dataset/pollution_dataset.csv")?;
//        features = normalize(&features);
        //let msg = run_classification();
//        self.result = msg.into();
//        self.result_changed();
//    }
//}

//fn main() {
  //  qml_register_type::<RustModel>(cstr!("RustModel"), 1, 0, cstr!("RustModel"));

//    let mut engine = QmlEngine::new();
  //  engine.load_file("main.qml".into()); // <- pastikan file ini ada di root project
    //engine.exec();
//}


// fungsi load data
fn load_data(path: &str) -> Result<(Array2<f64>, Array1<usize>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;

    let mut records: Vec<_> = rdr.records().collect::<Result<_, _>>()?;
    records.shuffle(&mut thread_rng());

    let mut features = Vec::new();
    let mut labels = Vec::new();

    for record in &records {
        let feat: Vec<f64> = (0..5)
            .map(|i| record[i].trim().parse::<f64>().unwrap_or(0.0))
            .collect();
        features.push(feat);

        let label = record[5].trim().parse::<usize>().unwrap_or(0);
        labels.push(label);
    }

    Ok((
        Array2::from_shape_vec((features.len(), 5), features.concat())?,
        Array1::from_vec(labels),
    ))
}

// fn normalize(features: &Array2<f64>) -> Array2<f64> {
//    let means = features.mean_axis(Axis(0)).unwrap();
//    let stds = features.std_axis(Axis(0), 0.0);
//
//    Array2::from_shape_fn(features.raw_dim(), |(i, j)| {
//        let std = stds[j].max(1e-8); // Hindari pembagian 0
//        (features[(i, j)] - means[j]) / std
//    })
//}

// =================== EUCLIDEAN DISTANCE ===================
fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

// knn
fn predict_knn(
    train: &Array2<f64>,
    train_labels: &Array1<usize>,
    test: &Array2<f64>,
    k: usize,
) -> Array1<usize> {
    let mut predictions = Array1::zeros(test.nrows());

    for (i, test_sample) in test.axis_iter(Axis(0)).enumerate() {
        let mut distances: Vec<_> = train
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(j, train_sample)| {
                let dist = euclidean_distance(&train_sample.to_owned(), &test_sample.to_owned());
                (j, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut counts = HashMap::new();
        for &(idx, _) in distances.iter().take(k) {
            let label = train_labels[idx];
            *counts.entry(label).or_insert(0) += 1;
        }

        let predicted_label = counts.into_iter().max_by_key(|&(_, c)| c).unwrap().0;
        predictions[i] = predicted_label;
    }

    predictions
}

// svm
fn train_svm(train_x: Array2<f64>, train_y: Array1<f64>, test_x: Array2<f64>) -> Array1<f64> {
    let dataset = Dataset::new(train_x, train_y);

    let model = Svm::params()
        .gaussian_kernel(100.0)
        .fit(&dataset)
        .expect("Gagal melatih SVM");

    model.predict(&test_x)
}

//gamma
// fn train_svm(train_x: Array2<f64>, train_y: Array1<f64>, test_x: Array2<f64>, gamma: f64) -> Array1<f64> {
//    let dataset = Dataset::new(train_x, train_y);
//
//    let model = Svm::params()
//        .gaussian_kernel(gamma)
//        .fit(&dataset)
//        .expect("Gagal melatih SVM");
//
//    model.predict(&test_x)
//}

// akurasi
fn accuracy(pred: &Array1<f64>, target: &Array1<usize>) -> f64 {
    pred.iter().zip(target.iter()).filter(|(p, t)| **p as usize == **t).count() as f64 / pred.len() as f64
}

// plot (fleksibel fitur)
fn plot_svm_neighbors(
    train: &Array2<f64>,
    train_labels: &Array1<usize>,
    test: &Array2<f64>,
    test_label: usize,
    file_name: &str,
    feat_idx: (usize, usize),
    feat_names: (&str, &str),
) -> Result<(), Box<dyn Error>> {
    let caption = format!(
        "SVM Decision Boundary (Predicted: {})",
        match test_label {
            0 => "Safe",
            1 => "Hazard",
            _ => "Unknown",
        }
    );

    let x_vals: Vec<f64> = train.column(feat_idx.0).iter().cloned().collect();
    let y_vals: Vec<f64> = train.column(feat_idx.1).iter().cloned().collect();
    let x_min = x_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let root = BitMapBackend::new(file_name, (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&caption, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc(format!("{} (Normalized)", feat_names.0))
        .y_desc(format!("{} (Normalized)", feat_names.1))
        .draw()?;

    let two_feature_train = Array2::from_shape_fn((train.nrows(), 2), |(i, j)| {
        match j {
            0 => train[[i, feat_idx.0]],
            1 => train[[i, feat_idx.1]],
            _ => 0.0,
        }
    });

    let dataset = Dataset::new(two_feature_train.clone(), train_labels.clone())
        .with_feature_names(vec![feat_names.0, feat_names.1]);

    let model = MultiLogisticRegression::default()
        .max_iterations(100)
        .fit(&dataset)?;

    let grid_size = 100;
    let x_step = (x_max - x_min) / grid_size as f64;
    let y_step = (y_max - y_min) / grid_size as f64;

    for i in 0..grid_size {
        for j in 0..grid_size {
            let x = x_min + i as f64 * x_step;
            let y = y_min + j as f64 * y_step;
            let point = Array2::from_shape_vec((1, 2), vec![x, y])?;
            let prediction = model.predict(&point)[0];

            let color = match prediction {
                0 => RGBColor(255, 200, 200),
                1 => RGBColor(200, 255, 200),
                _ => WHITE,
            };

            chart.draw_series(std::iter::once(Rectangle::new(
                [(x, y), (x + x_step, y + y_step)],
                ShapeStyle::from(&color).filled(),
            )))?;
        }
    }

    for (i, label) in train_labels.iter().enumerate() {
        let color = match label {
            0 => RED.mix(0.7),
            1 => GREEN.mix(0.7),
            _ => BLACK.into(),
        };

        chart.draw_series(PointSeries::of_element(
            vec![(train[[i, feat_idx.0]], train[[i, feat_idx.1]])],
            8,
            ShapeStyle::from(&color).filled(),
            &|coord, size, style| {
                EmptyElement::at(coord)
                    + Circle::new((0, 0), size, style)
                    + Text::new(
                        match label {
                            0 => "U",
                            1 => "S",
                            _ => "?",
                        },
                        (0, 10),
                        ("sans-serif", 15).into_font(),
                    )
            },
        ))?;
    }

    let test_point = Array2::from_shape_vec(
        (1, 2),
        vec![test[[0, feat_idx.0]], test[[0, feat_idx.1]]],
    )?;

    let test_color = match test_label {
        0 => RED,
        1 => GREEN,
        _ => BLACK,
    };

    chart.draw_series(PointSeries::of_element(
        vec![(test_point[[0, 0]], test_point[[0, 1]])],
        15,
        ShapeStyle::from(&test_color).filled(),
        &|coord, size, style| {
            EmptyElement::at(coord)
                + Circle::new((0, 0), size, style)
                + Text::new("X", (0, 15), ("sans-serif", 20).into_font())
        },
    ))?;

    root.present()?;
    println!("📊 Plot disimpan di '{}'", file_name);
    Ok(())
}

// main
fn main() -> Result<(), Box<dyn Error>> {
    let (features, labels_usize) = load_data("dataset/pollution_dataset_realistic_status.csv")?;
    //let (mut features, labels_usize) = load_data("dataset/pollution_dataset.csv")?;
    //features = normalize(&features);

    let label_count = labels_usize.iter().fold([0; 2], |mut acc, &x| {
        acc[x] += 1;
        acc
    });
    println!("📊 Distribusi label: Safe = {}, Hazard = {}", label_count[0], label_count[1]);

    let split_idx = (features.nrows() as f64 * 0.8) as usize;
    let (train_x, test_x) = features.view().split_at(Axis(0), split_idx);
    let (train_y_usize, test_y_usize) = labels_usize.view().split_at(Axis(0), split_idx);
    let train_y_f64 = train_y_usize.to_owned().mapv(|x| x as f64);

    let svm_preds = train_svm(train_x.to_owned(), train_y_f64.clone(), test_x.to_owned());
    let svm_acc = accuracy(&svm_preds, &test_y_usize.to_owned());
    println!("🎯 Akurasi SVM: {:.2}%", svm_acc * 100.0);

    //let mut best_gamma = 0.0;
    //let mut best_acc = 0.0;
    //let mut best_preds = Array1::zeros(test_y_usize.len());
    
    // println!("📈 Grid Search Gamma SVM:");
    // for &gamma in &[0.01, 0.1, 1.0, 10.0, 100.0] {
    //    let preds = train_svm(train_x.to_owned(), train_y_f64.clone(), test_x.to_owned(), gamma);
    //    let acc = accuracy(&preds, &test_y_usize.to_owned());
    //    println!("  - Gamma = {:>6} → Akurasi = {:.2}%", gamma, acc * 100.0);
        
    //    if acc > best_acc {
    //        best_acc = acc;
    //        best_gamma = gamma;
    //        best_preds = preds;
    //    }
    // }
    
    // println!("✅ Gamma terbaik: {} → Akurasi = {:.2}%", best_gamma, best_acc * 100.0);
    
    let knn_preds = predict_knn(&train_x.to_owned(), &train_y_usize.to_owned(), &test_x.to_owned(), 3);
    let knn_acc = knn_preds
        .iter()
        .zip(test_y_usize.iter())
        .filter(|(p, t)| *p == *t)
        .count() as f64 / knn_preds.len() as f64;
    println!("🎯 Akurasi KNN: {:.2}%", knn_acc * 100.0);

    println!("\n🔍 20 Prediksi SVM vs Aktual:");
    for (i, (pred, actual)) in svm_preds.iter().zip(test_y_usize.iter()).take(20).enumerate() {
    //for (i, (pred, actual)) in best_preds.iter().zip(test_y_usize.iter()).take(20).enumerate() {
        println!("Data {:2}: Prediksi = {}, Aktual = {}", i + 1, pred, actual);
    }

    println!("\n🔍 20 Prediksi KNN vs Aktual:");
    for (i, (pred, actual)) in knn_preds.iter().zip(test_y_usize.iter()).take(20).enumerate() {
        println!("Data {:2}: Prediksi = {}, Aktual = {}", i + 1, pred, actual);
    }

    fs::create_dir_all("result")?;

    plot_svm_neighbors(
        &train_x.to_owned(),
        &train_y_usize.to_owned(),
        &test_x.to_owned(),
        svm_preds[0] as usize,
        //best_preds[0] as usize,
        "result/pm10_vs_coawawoo.png",
        (2, 4),
        ("PM10", "CO")
    )?;

    Ok(())
}
