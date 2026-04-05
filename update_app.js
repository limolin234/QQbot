const fs = require('fs');

let code = fs.readFileSync('tools/config-studio-web/src/App.tsx', 'utf-8');

// replace <h2>Scheduler 设置</h2>
code = code.replace(
    /<h2>Scheduler 设置<\/h2>/,
    `<h2 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>Scheduler 设置<button className="icon-btn" onClick={() => setOpenSchedulerHelp(true)} title="使用说明" style={{ borderRadius: '50%', width: '24px', height: '24px', padding: 0, minWidth: 0, lineHeight: '24px', display: 'inline-block', textAlign: 'center', fontSize: '14px', cursor: 'pointer' }}>?</button></h2>`
);

// remove <details className="scheduler-help top-gap" open>...</details>
const helpStart = code.indexOf('<details className="scheduler-help top-gap" open>');
const helpEnd = code.indexOf('</details>', helpStart) + '</details>'.length;
const helpHtml = code.substring(helpStart, helpEnd);
code = code.substring(0, helpStart) + code.substring(helpEnd);

// find where to put the modal
const panelEnd = code.lastIndexOf('</section>');
code = code.substring(0, panelEnd) + `
                        <FormModal open={openSchedulerHelp} title="Scheduler 功能使用说明" onClose={() => setOpenSchedulerHelp(false)}>
                            <div className="scheduler-help-content" style={{ maxHeight: '600px', overflowY: 'auto' }}>
${helpHtml.replace(/<details[^>]*>/, '').replace(/<summary>.*?<\/summary>/, '').replace(/<\/details>/, '')}
                            </div>
                        </FormModal>
` + code.substring(panelEnd);

fs.writeFileSync('tools/config-studio-web/src/App.tsx', code);
