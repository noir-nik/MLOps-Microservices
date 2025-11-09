<script lang="ts">
	import { onMount } from 'svelte';
	import axios from 'axios';
	import DatasetList from '../lib/components/DatasetList.svelte';
	import ModelList from '../lib/components/ModelList.svelte';
	import RunList from '../lib/components/RunList.svelte';
	import CollectScenario from '../lib/components/CollectScenario.svelte';
	import TrainScenario from '../lib/components/TrainScenario.svelte';
	import InferScenario from '../lib/components/InferScenario.svelte';
	import ReportModal from '../lib/components/ReportModal.svelte';

	const API_URL = import.meta.env.VITE_WEB_MASTER_API || 'http://localhost:8000';

	let datasets = $state<any[]>([]);
	let models = $state<any[]>([]);
	let runs = $state<any[]>([]);
	let health = $state<any>({});
	let activeTab = $state('collect');
	let selectedModel = $state<any>(null);
	let showReport = $state(false);

	async function fetchData() {
		try {
			const [datasetsRes, modelsRes, runsRes, healthRes] = await Promise.all([
				axios.get(`${API_URL}/datasets`),
				axios.get(`${API_URL}/models`),
				axios.get(`${API_URL}/runs`),
				axios.get(`${API_URL}/health`),
			]);

			datasets = datasetsRes.data;
			models = modelsRes.data;
			runs = runsRes.data.sort((a: any, b: any) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
			health = healthRes.data;
		} catch (error) {
			console.error('Error fetching data:', error);
		}
	}

	function openReport(model: any) {
		selectedModel = model;
		showReport = true;
	}

	onMount(() => {
		fetchData();
		const interval = setInterval(fetchData, 5000);
		return () => clearInterval(interval);
	});
</script>

<div class="bg-gray-50 min-h-screen">
	<header class="bg-white shadow">
		<div class="mx-auto px-4 py-6 max-w-7xl">
			<div class="flex justify-between items-center">
				<div>
					<h1 class="font-bold text-gray-900 text-3xl">MLOps Microservices</h1>
					<p class="mt-1 text-gray-600 text-sm">{health.version}</p>
				</div>
				<div class="flex gap-3">
					{#each Object.entries(health.services || {}) as [service, status]}
						<div class="flex items-center gap-2 bg-gray-100 px-3 py-1 rounded-full">
							<div class="w-2 h-2 rounded-full {status === 'healthy' ? 'bg-green-500' : 'bg-red-500'}"></div>
							<span class="font-medium text-gray-700 text-xs">{service}</span>
						</div>
					{/each}
				</div>
			</div>
		</div>
	</header>

	<main class="mx-auto px-4 py-8 max-w-7xl">
		<!-- Scenarios Section -->
		<section class="mb-8">
			<h2 class="mb-4 font-bold text-gray-900 text-2xl">Scenarios</h2>

			<div class="flex gap-2 mb-4 border-gray-200 border-b">
				<button
					class="px-4 py-2 font-medium transition-colors {activeTab === 'collect'
						? 'text-blue-600 border-b-2 border-blue-600'
						: 'text-gray-600 hover:text-gray-900'}"
					onclick={() => (activeTab = 'collect')}
				>
					1. Collect & Store
				</button>
				<button
					class="px-4 py-2 font-medium transition-colors {activeTab === 'train'
						? 'text-blue-600 border-b-2 border-blue-600'
						: 'text-gray-600 hover:text-gray-900'}"
					onclick={() => (activeTab = 'train')}
				>
					2. Train Model
				</button>
				<button
					class="px-4 py-2 font-medium transition-colors {activeTab === 'infer'
						? 'text-blue-600 border-b-2 border-blue-600'
						: 'text-gray-600 hover:text-gray-900'}"
					onclick={() => (activeTab = 'infer')}
				>
					3. Infer Model
				</button>
			</div>

			<div class="bg-white shadow p-6 rounded-lg">
				{#if activeTab === 'collect'}
					<CollectScenario {API_URL} onComplete={fetchData} />
				{:else if activeTab === 'train'}
					<TrainScenario {API_URL} {datasets} onComplete={fetchData} />
				{:else if activeTab === 'infer'}
					<InferScenario {API_URL} {models} onComplete={fetchData} />
				{/if}
			</div>
		</section>

		<!-- Data Lists Section -->
		<section class="gap-6 grid grid-cols-1 lg:grid-cols-3">
			<DatasetList {datasets} {API_URL} />
			<ModelList {models} {API_URL} {openReport} />
			<RunList {runs} />
		</section>
	</main>
</div>

{#if showReport && selectedModel}
	<ReportModal {API_URL} model={selectedModel} onClose={() => (showReport = false)} />
{/if}
