<script lang="ts">
	import axios from 'axios';

	interface Props {
		API_URL: string;
		onComplete: () => void;
	}

	let { API_URL, onComplete }: Props = $props();

	let batchSize = $state(100);
	let loading = $state(false);
	let result = $state<any>(null);
	let error = $state<string | null>(null);

	async function handleCollect() {
		loading = true;
		error = null;
		result = null;

		try {
			const response = await axios.post(`${API_URL}/scenarios/collect-and-store`, {
				batch_size: batchSize,
			});

			result = response.data;
			onComplete();
		} catch (err: any) {
			error = err.response?.data?.detail || err.message;
		} finally {
			loading = false;
		}
	}
</script>

<div>
	<h3 class="mb-4 font-semibold text-gray-900 text-xl">Collect and Store Dataset</h3>
	<p class="mb-6 text-gray-600">Fetch a batch of data from the collector and save it to storage.</p>

	<div class="space-y-4">
		<div>
			<label class="block mb-2 font-medium text-gray-700 text-sm">Batch Size</label>
			<input
				type="number"
				bind:value={batchSize}
				min="1"
				max="1000"
				class="px-3 py-2 border border-gray-300 focus:border-blue-500 rounded-md focus:ring-blue-500 w-full"
			/>
			<p class="mt-1 text-gray-500 text-xs">Number of rows to fetch from the Titanic dataset</p>
		</div>

		<button
			onclick={handleCollect}
			disabled={loading}
			class="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 px-4 py-2 rounded-md w-full font-medium text-white transition-colors disabled:cursor-not-allowed"
		>
			{loading ? 'Collecting...' : 'Collect & Store'}
		</button>

		{#if result}
			<div class="bg-green-50 p-4 border border-green-200 rounded-md">
				<h4 class="mb-2 font-semibold text-green-900">✓ Success</h4>
				<p class="mb-2 text-green-800 text-sm">{result.message}</p>
				<div class="text-green-700 text-sm">
					<p><strong>Dataset:</strong> {result.dataset_name}</p>
					<p><strong>Rows:</strong> {result.rows}</p>
					<p><strong>Columns:</strong> {result.columns}</p>
				</div>
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
