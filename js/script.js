/**
 * Função que busca o array de objetos dos carros e transforma em json.
 * pegando somente o importante para o exercicio e filtrando dados que nao precisarão ser usados.
*/
async function getData() {
  const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
  const carsData = await carsDataReq.json();
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower,
  }))
    .filter(car => (car.mpg != null && car.horsepower != null));
  console.log(cleaned)
  return cleaned;
}

// Plota os dados dos carros buscados
function plotData(data) {
  // Carregando e plotando os dados de entrada originais nos quais vamos treinar.
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  // Função que renderizará o gráfico na página
  tfvis.render.scatterplot(
    { name: 'Horsepower v MPG' },
    { values },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );
}

/**
 * A conversão será feita em três etapas : Embaralhamento, Conversão para Tensor  e normalização.
 * Tensor é a estrutura de dados primária usada no tensorflow.
*/
function convertToTensor(data) {

  return tf.tidy(() => {

    // Etapa 1. Embaralha os dados (array de objetos)
    tf.util.shuffle(data);

    // Etapa 2. Converte os dados to Tensor
    const inputs = data.map(d => d.horsepower)
    const labels = data.map(d => d.mpg); // labels = outputs

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Etapa 3. Normalize the data to the range 0 - 1 using min-max scaling

    /*
    A normalização é importante porque as partes internas de muitos modelos de aprendizado de máquina que você criará com o tensorflow.js foram projetadas para trabalhar com números que não são muito grandes.

    O objetivo da normalização é transformar os recursos em uma escala semelhante. Isso melhora o desempenho e a estabilidade do treinamento do modelo.
    */
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Retornando os limites min/max para serem usados no futuro
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}

// Treina o modelo, passando as entradas e saídas esperadas (Aprendizado supervisionado)
async function trainModel(model, inputs, labels) {
  // Prepare the model for training.  
  model.compile({
    optimizer: tf.train.adam(), // Algoritmo de otimização do modelo, faz a atualizacao
    loss: tf.losses.meanSquaredError, // Funcao de perda, com erroQuadraticoMédio, bom para eq lineares
    metrics: ['mse'],
  });

  const batchSize = 32;
  const epochs = 50; // Número de vezes que o modelo examinará todo o conjunto de dados fornecidos.

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

// Testa o modelo, com os dados normalizados, usando funções de predição
function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling 
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {

    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));

    const unNormXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin);

    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });


  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] }
  });

  const originalPoints = inputData.map(d => ({
    x: d.horsepower, y: d.mpg,
  }));


  tfvis.render.scatterplot(
    { name: 'Model Predictions vs Original Data' },
    { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );
}

// A arquitetura define quais funções que o modelo executará quando estiver rodando
function createModel() {
  // Cria um modelo sequencial
  const model = tf.sequential();

  // Adiciona uma única camada oculta
  // Dense porque em cada node das camadas estarão conectados entre si
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

  // Adiciona a camada de output
  model.add(tf.layers.dense({ units: 1, useBias: true }));

  return model;
}



async function run() {

  // Busca dados dos carros
  const data = await getData();
  plotData(data);

  // Instacia o modelo, passando para a biblioteca tfvis renderizar
  const model = createModel();
  tfvis.show.modelSummary({ name: 'Model Summary' }, model);

  // Converte os dados para uma forma que o tensorFlow entende internamente
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;
  await trainModel(model, inputs, labels);

  // Testa o modelo
  testModel(model, data, tensorData);
}


document.addEventListener('DOMContentLoaded', run);