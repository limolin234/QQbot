const fs = require('fs');
let code = fs.readFileSync('src/App.tsx', 'utf-8');

const target = `                                        />
                                    </div>
                                </FormModal>
                            </div>
                        ) : (`;

const replacement = `                                        />
                                    </div>
                                </FormModal>
                            </>
                        ) : (`;

code = code.replace(target, replacement);
fs.writeFileSync('src/App.tsx', code);
