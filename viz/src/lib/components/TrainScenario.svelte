<script lang="ts">
	import axios from 'axios';

	interface Props {
		API_URL: string;
		datasets: any[];
		onComplete: () => void;
	}

	let { API_URL, datasets, onComplete }: Props = $props();

	let selectedDataset = $state<number | null>(null);
	let modelType = $state('LogisticRegression');
	let hyperparameters = $state<any>({
		LogisticRegression: { max_iter: 1000, C: 1.0 },
		RandomForest: { n_estimators: 100, max_depth: 10 },
		KNN: { n_neighbors: 5, weights: 'uniform' },
	});
	let loading = $state(false);
	let result = $state<any>(null);
	let error = $state<string | null>(null);
	let progress = $state<string>('');
	let eventSource = $state<EventSource | null>(null);

	async function handleTrain() {
		if (!selectedDataset) {
			error = 'Please select a dataset';
			return;
		}

		loading = true;
		error = null;
		result = null;
		progress = '';

		try {
			const response = await axios.post(`${API_URL}/scenarios/train-model`, {
				dataset_id: selectedDataset,
				model_type: modelType,
				hyperparameters: hyperparameters[modelType],
			});

			const runId = response.data.run_id;

			// Connect to progress stream
			eventSource = new EventSource(`${API_URL}/progress/${runId}`);

			eventSource.onmessage = (event) => {
				try {
					const data = JSON.parse(event.data);
					if (data.message) {
						progress = `${data.progress?.toFixed(0) || 0}% - ${data.message}`;
					}
					if (data.progress >= 100) {
						eventSource?.close();
						setTimeout(() => {
							result = response.data;
							loading = false;
							onComplete();
						}, 1000);
					}
				} catch (e) {
					console.error('Error parsing progress:', e);
				}
			};

			eventSource.onerror = () => {
				eventSource?.close();
				loading = false;
			};
		} catch (err: any) {
			error = err.response?.data?.detail || err.message;
			loading = false;
		}
	}

	function updateHyperparameter(key: string, value: any) {
		hyperparameters[modelType][key] = value;
	}
</script>

<div>
	<h3 class="mb-4 font-semibold text-gray-900 text-xl">Train Model</h3>
	<p class="mb-6 text-gray-600">Select a dataset and model type to train a machine learning model.</p>

	<div class="space-y-4">
		<div>
			<label class="block mb-2 font-medium text-gray-700 text-sm">Dataset</label>
			<select bind:value={selectedDataset} class="px-3 py-2 border border-gray-300 focus:border-blue-500 rounded-md focus:ring-blue-500 w-full">
				<option value={null}>Select a dataset...</option>
				{#each datasets as dataset}
					<option value={dataset.id}>
						{dataset.name} ({dataset.rows} rows)
					</option>
				{/each}
			</select>
		</div>

		<div>
			<label class="block mb-2 font-medium text-gray-700 text-sm">Model Type</label>
			<select bind:value={modelType} class="px-3 py-2 border border-gray-300 focus:border-blue-500 rounded-md focus:ring-blue-500 w-full">
				<option value="LogisticRegression">Logistic Regression</option>
				<option value="RandomForest">Random Forest</option>
				<option value="KNN">K-Nearest Neighbors</option>
			</select>
		</div>

		<div class="bg-gray-50 p-4 rounded-md">
			<h4 class="mb-3 font-semibold text-gray-900 text-sm">Hyperparameters</h4>
			{#if modelType === 'LogisticRegression'}
				<div class="space-y-3">
					<div>
						<label class="block mb-1 text-gray-600 text-xs">Max Iterations</label>
						<input
							type="number"
							value={hyperparameters.LogisticRegression.max_iter}
							oninput={(e) => updateHyperparameter('max_iter', parseInt(e.currentTarget.value))}
							class="px-2 py-1 border border-gray-300 rounded w-full text-sm"
						/>
					</div>
					<div>
						<label class="block mb-1 text-gray-600 text-xs">C (Regularization)</label>
						<input
							type="number"
							step="0.1"
							value={hyperparameters.LogisticRegression.C}
							oninput={(e) => updateHyperparameter('C', parseFloat(e.currentTarget.value))}
							class="px-2 py-1 border border-gray-300 rounded w-full text-sm"
						/>
					</div>
				</div>
			{:else if modelType === 'RandomForest'}
				<div class="space-y-3">
					<div>
						<label class="block mb-1 text-gray-600 text-xs">Number of Estimators</label>
						<input
							type="number"
							value={hyperparameters.RandomForest.n_estimators}
							oninput={(e) => updateHyperparameter('n_estimators', parseInt(e.currentTarget.value))}
							class="px-2 py-1 border border-gray-300 rounded w-full text-sm"
						/>
					</div>
					<div>
						<label class="block mb-1 text-gray-600 text-xs">Max Depth</label>
						<input
							type="number"
							value={hyperparameters.RandomForest.max_depth}
							oninput={(e) => updateHyperparameter('max_depth', parseInt(e.currentTarget.value))}
							class="px-2 py-1 border border-gray-300 rounded w-full text-sm"
						/>
					</div>
				</div>
			{:else if modelType === 'KNN'}
				<div class="space-y-3">
					<div>
						<label class="block mb-1 text-gray-600 text-xs">Number of Neighbors</label>
						<input
							type="number"
							value={hyperparameters.KNN.n_neighbors}
							oninput={(e) => updateHyperparameter('n_neighbors', parseInt(e.currentTarget.value))}
							class="px-2 py-1 border border-gray-300 rounded w-full text-sm"
						/>
					</div>
					<div>
						<label class="block mb-1 text-gray-600 text-xs">Weights</label>
						<select
							value={hyperparameters.KNN.weights}
							onchange={(e) => updateHyperparameter('weights', e.currentTarget.value)}
							class="px-2 py-1 border border-gray-300 rounded w-full text-sm"
						>
							<option value="uniform">Uniform</option>
							<option value="distance">Distance</option>
						</select>
					</div>
				</div>
			{/if}
		</div>

		<button
			onclick={handleTrain}
			disabled={loading || !selectedDataset}
			class="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 px-4 py-2 rounded-md w-full font-medium text-white transition-colors disabled:cursor-not-allowed"
		>
			{loading ? 'Training...' : 'Train Model'}
		</button>

		{#if progress}
			<div class="bg-blue-50 p-4 border border-blue-200 rounded-md">
				<div class="flex items-center gap-2 mb-2">
					<div class="border-2 border-t-transparent border-blue-600 rounded-full w-4 h-4 animate-spin"></div>
					<h4 class="font-semibold text-blue-900">Training in Progress</h4>
				</div>
				<p class="text-blue-800 text-sm">{progress}</p>
			</div>
		{/if}

		{#if result}
			<div class="bg-green-50 p-4 border border-green-200 rounded-md">
				<h4 class="mb-2 font-semibold text-green-900">✓ Training Started</h4>
				<p class="text-green-800 text-sm">{result.message}</p>
				<p class="mt-1 text-green-700 text-xs">Run ID: {result.run_id}</p>
			</div>
		{/if}

		{#if error}
			<div class="bg-red-50 p-4 border border-red-200 rounded-md">
				<h4 class="mb-2 font-semibold text-red-900">✗ Error</h4>
				<p class="text-red-800 text-sm">{error}</p>
			</div>
		{/if}
	</div>
</div>
