// SentimemeNet Frontend JavaScript
// Handles image upload, analysis, and results display

document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 SentimemeNet script loaded - v2');
    
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    
    console.log('✅ DOM elements loaded:', {
        dropZone: !!dropZone,
        fileInput: !!fileInput,
        imagePreview: !!imagePreview
    });
    const previewImage = document.getElementById('previewImage');
    const removeImage = document.getElementById('removeImage');
    const memeText = document.getElementById('memeText');
    const charCount = document.getElementById('charCount');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const analyzeBtnText = document.getElementById('analyzeBtnText');
    const sampleMeme = document.getElementById('sampleMeme');
    const resultsSection = document.getElementById('resultsSection');
    const emptyResults = document.getElementById('emptyResults');
    const exportBtn = document.getElementById('exportBtn');
    const processingTime = document.getElementById('processingTime');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressPercent = document.getElementById('progressPercent');
    const progressText = document.getElementById('progressText');
    const analysisCardsGrid = document.getElementById('analysisCardsGrid');
    const nonMemeMessage = document.getElementById('nonMemeMessage');
    
    // State
    let currentImageData = null;
    let analysisResults = null;

    // Event Listeners
    dropZone.addEventListener('click', (e) => {
        console.log('🖱️ Drop zone clicked');
        console.log('fileInput element:', fileInput);
        if (fileInput) {
            console.log('Triggering fileInput.click()');
            fileInput.click();
        } else {
            console.error('❌ fileInput is null!');
        }
    });
    
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    dropZone.addEventListener('dragenter', handleDragOver);
    fileInput.addEventListener('change', handleFileSelect);
    removeImage.addEventListener('click', clearImage);
    memeText.addEventListener('input', updateCharCount);
    analyzeBtn.addEventListener('click', analyzeMeme);
    sampleMeme.addEventListener('click', loadSampleMeme);
    exportBtn.addEventListener('click', exportResults);
    
    console.log('✅ All event listeners attached successfully');

    // Clipboard paste event
    document.addEventListener('paste', handlePaste);

    // Prevent default drag behavior on the entire document
    document.addEventListener('dragover', (e) => {
        e.preventDefault();
    });
    
    document.addEventListener('drop', (e) => {
        if (!dropZone.contains(e.target)) {
            e.preventDefault();
        }
    });

    // Functions
    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('drag-over');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');
    }

    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length) {
            handleFiles(files[0]);
        }
    }

    function handleFileSelect(e) {
        console.log('📁 File selected:', e.target.files);
        if (e.target.files.length) {
            handleFiles(e.target.files[0]);
        }
    }

    function handlePaste(e) {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const blob = items[i].getAsFile();
                handleFiles(blob);
                break;
            }
        }
    }

    function handleFiles(file) {
        console.log('📂 handleFiles called with:', file);
        
        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
        if (!validTypes.includes(file.type)) {
            console.error('❌ Invalid file type:', file.type);
            alert('Please select a valid image file (JPEG, PNG, or GIF)');
            return;
        }
        
        // Validate file size (10MB max)
        if (file.size > 10 * 1024 * 1024) {
            console.error('❌ File too large:', file.size);
            alert('File size should be less than 10MB');
            return;
        }

        console.log('✅ File validation passed');
        
        // Preview image
        const reader = new FileReader();
        reader.onload = function(e) {
            console.log('✅ File read complete');
            currentImageData = e.target.result;
            previewImage.src = currentImageData;
            imagePreview.classList.remove('hidden');
            analyzeBtn.disabled = false;
            
            // Reset results and hide progress
            resultsSection.classList.add('hidden');
            emptyResults.classList.remove('hidden');
            progressContainer.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }
    function clearImage() {
        fileInput.value = '';
        previewImage.src = '';
        currentImageData = null;
        imagePreview.classList.add('hidden');
        analyzeBtn.disabled = true;
        resultsSection.classList.add('hidden');
        emptyResults.classList.remove('hidden');
        progressContainer.classList.add('hidden');
        analysisCardsGrid.classList.remove('hidden');
        nonMemeMessage.classList.add('hidden');
    }

    function updateCharCount() {
        const count = memeText.value.length;
        charCount.textContent = count;
        if (count > 500) {
            charCount.classList.add('text-red-500');
            memeText.value = memeText.value.substring(0, 500);
        } else {
            charCount.classList.remove('text-red-500');
        }
    }

    async function analyzeMeme() {
        if (!currentImageData) {
            alert('Please upload an image first');
            return;
        }

        // Show loading state
        setLoadingState(true);

        // Show results section
        resultsSection.classList.remove('hidden');
        emptyResults.classList.add('hidden');

        // Reset all cards to analyzing state
        resetAnalysisCards();

        const startTime = Date.now();

        try {
            // Simulate progress updates
            updateProgress(10, 'Preparing image data...');
            await new Promise(resolve => setTimeout(resolve, 200));

            // Prepare request data
            const requestData = {
                image: currentImageData.split(',')[1], // Remove data:image/xxx;base64, prefix
                ocr_text: memeText.value.trim() || null
            };

            console.log('Sending request to backend...');
            updateProgress(20, 'Sending to server...');

            // Make API call
            const response = await fetch('/api/analyze_meme', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            console.log('Response received:', response.status);
            updateProgress(30, 'Processing meme detection...');

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server error:', errorText);
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            updateProgress(50, 'Running sentiment analysis...');
            await new Promise(resolve => setTimeout(resolve, 300));

            const data = await response.json();
            console.log('Analysis results:', data);
            
            const endTime = Date.now();
            const duration = ((endTime - startTime) / 1000).toFixed(1);

            if (data.success) {
                updateProgress(100, 'Complete!');
                analysisResults = data.results;
                analysisResults.processing_time = `${duration}s`;
                displayResults(data);
            } else {
                throw new Error(data.error || 'Analysis failed');
            }

        } catch (error) {
            console.error('Analysis error:', error);
            alert('Analysis failed: ' + error.message + '\n\nPlease check:\n1. Backend server is running\n2. You opened the page from http://localhost:5000 (not file://)\n3. Browser console for more details');
            
            // Reset UI
            resultsSection.classList.add('hidden');
            emptyResults.classList.remove('hidden');
            progressContainer.classList.add('hidden');
        } finally {
            setLoadingState(false);
        }
    }

    function updateProgress(percent, message) {
        progressBar.style.width = `${percent}%`;
        progressPercent.textContent = `${Math.round(percent)}%`;
        progressText.textContent = message;
    }

    function setLoadingState(isLoading) {
        analyzeBtn.disabled = isLoading;
        if (isLoading) {
            analyzeBtnText.textContent = 'Analyzing...';
            progressContainer.classList.remove('hidden');
            updateProgress(0, 'Initializing analysis...');
        } else {
            analyzeBtnText.textContent = 'Analyze Meme';
            progressContainer.classList.add('hidden');
        }
    }

    function resetAnalysisCards() {
        // Reset primary result
        document.getElementById('memeLabel').textContent = 'Analyzing...';
        document.getElementById('memeConfidence').textContent = '--%';
        document.getElementById('memeConfidenceBar').style.width = '0%';

        // Reset all analysis cards
        const cards = document.querySelectorAll('.analysis-card');
        cards.forEach(card => {
            const status = card.querySelector('.status');
            const progress = card.querySelector('.progress');
            const confidenceText = card.querySelector('.confidence-text');
            
            status.textContent = 'Analyzing...';
            status.className = 'status';
            progress.style.width = '0%';
            confidenceText.textContent = '--% confidence';
        });
    }

    function displayResults(data) {
        if (!data.success) {
            alert('Analysis failed: ' + data.error);
            return;
        }

        const results = data.results;
        
        // Animate results reveal
        setTimeout(() => {
            // Update primary result
            updatePrimaryResult(results.meme_detection);
            
            // Check if it's a meme or not
            if (results.meme_detection.is_meme) {
                // Show sentiment cards, hide non-meme message
                analysisCardsGrid.classList.remove('hidden');
                nonMemeMessage.classList.add('hidden');
                
                // Update analysis cards with staggered animation
                const categories = ['humour', 'motivational', 'offensive', 'sarcasm', 'sentiment'];
                categories.forEach((category, index) => {
                    setTimeout(() => {
                        updateAnalysisCard(category, results[category]);
                    }, index * 100);
                });
            } else {
                // Hide sentiment cards, show non-meme message
                analysisCardsGrid.classList.add('hidden');
                nonMemeMessage.classList.remove('hidden');
            }
            
            // Update processing time
            processingTime.textContent = `Processing time: ${results.processing_time}`;
        }, 100);
    }

    function updatePrimaryResult(result) {
        const badge = document.getElementById('memeBadge');
        const badgeIcon = badge.querySelector('i');
        const label = document.getElementById('memeLabel');
        const confidence = document.getElementById('memeConfidence');
        const confidenceBar = document.getElementById('memeConfidenceBar');
        
        // Update badge styling
        badge.className = 'w-10 h-10 rounded-full flex items-center justify-center mr-3';
        badgeIcon.className = 'w-5 h-5';
        
        if (result.is_meme) {
            badge.classList.add('bg-green-100');
            badgeIcon.classList.add('text-green-500');
            confidenceBar.classList.remove('bg-gray-400');
            confidenceBar.classList.add('bg-green-500');
        } else {
            badge.classList.add('bg-red-100');
            badgeIcon.classList.add('text-red-500');
            confidenceBar.classList.remove('bg-gray-400');
            confidenceBar.classList.add('bg-red-500');
        }
        
        // Update text
        label.textContent = result.label;
        confidence.textContent = `${(result.confidence * 100).toFixed(1)}%`;
        
        // Animate confidence bar
        setTimeout(() => {
            confidenceBar.style.width = `${result.confidence * 100}%`;
        }, 50);
    }

    function updateAnalysisCard(category, result) {
        const card = document.querySelector(`.analysis-card[data-category="${category}"]`);
        const status = card.querySelector('.status');
        const progress = card.querySelector('.progress');
        const confidenceText = card.querySelector('.confidence-text');
        
        // Set status text and color
        status.textContent = result.label;
        
        // Apply color classes based on result
        if (category === 'offensive') {
            status.className = result.prediction ? 'status status-negative' : 'status status-positive';
        } else {
            status.className = result.prediction ? 'status status-positive' : 'status status-neutral';
        }
        
        // Update progress bar with animation
        setTimeout(() => {
            progress.style.width = `${result.confidence * 100}%`;
        }, 50);
        
        // Update confidence text
        confidenceText.textContent = `${(result.confidence * 100).toFixed(1)}% confidence`;
        
        // Add fade-in animation
        card.classList.add('fade-in');
    }

    function loadSampleMeme() {
        // Create a sample meme using a placeholder service
        const sampleUrl = 'https://via.placeholder.com/400x400/4A90E2/ffffff?text=Sample+Meme';
        
        fetch(sampleUrl)
            .then(response => response.blob())
            .then(blob => {
                const reader = new FileReader();
                reader.onload = function(e) {
                    currentImageData = e.target.result;
                    previewImage.src = currentImageData;
                    imagePreview.classList.remove('hidden');
                    analyzeBtn.disabled = false;
                    
                    // Set sample text
                    memeText.value = "When you finally understand recursion after the 10th explanation";
                    updateCharCount();
                    
                    // Reset results
                    resultsSection.classList.add('hidden');
                    emptyResults.classList.remove('hidden');
                };
                reader.readAsDataURL(blob);
            })
            .catch(error => {
                console.error('Error loading sample meme:', error);
                alert('Could not load sample meme. Please upload your own image.');
            });
    }

    function exportResults() {
        if (!analysisResults) {
            alert('No analysis results to export');
            return;
        }

        // Create exportable data
        const exportData = {
            timestamp: new Date().toISOString(),
            image_name: 'meme_analysis',
            ocr_text: analysisResults.ocr_text || memeText.value,
            results: {
                meme_detection: analysisResults.meme_detection,
                humour: analysisResults.humour,
                motivational: analysisResults.motivational,
                offensive: analysisResults.offensive,
                sarcasm: analysisResults.sarcasm,
                sentiment: analysisResults.sentiment
            },
            processing_time: analysisResults.processing_time
        };

        // Convert to JSON and download
        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `sentimemenet_analysis_${Date.now()}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    function generateMockResults() {
        // Generate mock results for demo purposes
        return {
            success: true,
            results: {
                meme_detection: {
                    is_meme: true,
                    confidence: 0.9253,
                    label: "Meme"
                },
                humour: {
                    prediction: true,
                    confidence: 0.5220,
                    label: "Funny"
                },
                motivational: {
                    prediction: false,
                    confidence: 0.6155,
                    label: "Not Motivational"
                },
                offensive: {
                    prediction: false,
                    confidence: 0.5858,
                    label: "Not Offensive"
                },
                sarcasm: {
                    prediction: true,
                    confidence: 0.5585,
                    label: "Sarcastic"
                },
                sentiment: {
                    prediction: true,
                    confidence: 0.5810,
                    label: "Positive"
                },
                ocr_text: memeText.value || "No text provided",
                processing_time: "0.0s"
            }
        };
    }
});
