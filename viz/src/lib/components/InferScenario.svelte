<script lang="ts">
  import axios from 'axios';
  
  interface Props {
    API_URL: string;
    models: any[];
    onComplete: () => void;
  }
  
  let { API_URL, models, onComplete }: Props = $props();
  
  let selectedModel = $state<number | null>(null);
  let inputMethod = $state<'text' | 'file'>('text');
  let csvText = $state('Pclass,Sex,Age,SibSp,Parch,Fare,Embarked\n3,1,22,1,0,7.25,2\n1,0,38,1,0,71.28,0');
  let file = $state<File | null>(null);
  let loading = $state(false);
  let result = $state<any>(null);
  let error = $state<string | null>(null);
  let progress = $state<string>('');
  let eventSource = $state<EventSource | null>(null);
  
  async function handleInfer() {
    if (!selectedModel) {
      error = 'Please select a model';
      return;
    }
    
    let data = '';
    
    if (inputMethod === 'file' && file) {
      data = await file.text();
    } else {
      data = csvText;
    }
    
    if (!data.trim()) {
      error = 'Please provide input data';
      return;
    }
    
    loading = true;
    error = null;
    result = null;
    progress = '';
    
    try {
      const response = await axios.post(`${API_URL}/scenarios/infer-model`, {
        model_id: selectedModel,
        data: data
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
  
  function handleFileChange(e: Event) {
    const target = e.target as HTMLInputElement;
    file = target.files?.[0] || null;
  }
</script>
<!-- svelte-ignore a11y_label_has_associated_control -->

<div>
  <h3 class="mb-4 font-semibold text-gray-900 text-xl">Infer Model</h3>
  <p class="mb-6 text-gray-600">Use a trained model to make predictions on new data.</p>
  
  <div class="space-y-4">
    <div>
      <label class="block mb-2 font-medium text-gray-700 text-sm">Model</label>
      <select 
        bind:value={selectedModel}
        class="px-3 py-2 border border-gray-300 focus:border-blue-500 rounded-md focus:ring-blue-500 w-full"
      >
        <option value={null}>Select a model...</option>
        {#each models as model}
          <option value={model.id}>
            {model.name} - {model.model_type} (Accuracy: {(model.accuracy * 100).toFixed(1)}%)
          </option>
        {/each}
      </select>
    </div>
    
    <div>
      <label class="block mb-2 font-medium text-gray-700 text-sm">Input Method</label>
      <div class="flex gap-4">
        <label class="flex items-center">
          <input 
            type="radio" 
            value="text"
            bind:group={inputMethod}
            class="mr-2"
          />
          <span class="text-sm">CSV Text</span>
        </label>
        <label class="flex items-center">
          <input 
            type="radio" 
            value="file"
            bind:group={inputMethod}
            class="mr-2"
          />
          <span class="text-sm">Upload File</span>
        </label>
      </div>
    </div>
    
    {#if inputMethod === 'text'}
      <div>
        <label class="block mb-2 font-medium text-gray-700 text-sm">CSV Data</label>
        <textarea 
          bind:value={csvText}
          rows="6"
          class="px-3 py-2 border border-gray-300 focus:border-blue-500 rounded-md focus:ring-blue-500 w-full font-mono text-sm"
          placeholder="Pclass,Sex,Age,SibSp,Parch,Fare,Embarked"
        ></textarea>
        <p class="mt-1 text-gray-500 text-xs">Enter CSV data with headers. Sex: 0=female, 1=male; Embarked: 0=S, 1=C, 2=Q</p>
      </div>
    {:else}
      <div>
        <label class="block mb-2 font-medium text-gray-700 text-sm">Upload CSV File</label>
        <input 
          type="file" 
          accept=".csv"
          onchange={handleFileChange}
          class="px-3 py-2 border border-gray-300 focus:border-blue-500 rounded-md focus:ring-blue-500 w-full"
        />
      </div>
    {/if}
    
    <button 
      onclick={handleInfer}
      disabled={loading || !selectedModel}
      class="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 px-4 py-2 rounded-md w-full font-medium text-white transition-colors disabled:cursor-not-allowed"
    >
      {loading ? 'Running Inference...' : 'Run Inference'}
    </button>
    
    {#if progress}
      <div class="bg-blue-50 p-4 border border-blue-200 rounded-md">
        <div class="flex items-center gap-2 mb-2">
          <div class="border-2 border-t-transparent border-blue-600 rounded-full w-4 h-4 animate-spin"></div>
          <h4 class="font-semibold text-blue-900">Inference in Progress</h4>
        </div>
        <p class="text-blue-800 text-sm">{progress}</p>
      </div>
    {/if}
    
    {#if result}
      <div class="bg-green-50 p-4 border border-green-200 rounded-md">
        <h4 class="mb-2 font-semibold text-green-900">✓ Inference Started</h4>
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
